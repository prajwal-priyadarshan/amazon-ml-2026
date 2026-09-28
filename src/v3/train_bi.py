"""Fine-tune the retrieval bi-encoder with InfoNCE over in-batch and mined hard negatives.

Why fine-tune at all: the dataset ships about 7.6M labelled matches with *systematic* noise
-- suffixes moved between fields, domains glued into names, native-script names against
romanised ones, house numbers off by one. A generic checkpoint has never seen that noise
distribution. Training on it is the difference between a view that adds recall and a view
that duplicates what the character n-grams already found.

The negatives matter more than the positives. In-batch negatives are nearly free but easy;
the mined ones come from the lexical retrieval's top hits that are *not* the true match,
which is exactly the "plausible but wrong" region where retrieval loses recall. Every
anchor therefore contributes one positive, ``hard`` mined negatives, and the whole rest of
the batch.

The encoder is initialised from an embedding-pretrained checkpoint, never a raw causal LM:
for bi-encoders that initialisation is consistently stronger.
"""
import math
import os

import numpy as np

from .dense import DEFAULT_MODEL, encode_batch, torch_device
from .paths import log


def train(out_dir, anchors, positives, negatives, cfg):
    """anchors/positives: (N,) text arrays. negatives: (N, H) text array (may be empty)."""
    import torch
    from torch.optim import AdamW
    from transformers import AutoModel, AutoTokenizer

    device = torch_device()
    tok = AutoTokenizer.from_pretrained(cfg.bi_model or DEFAULT_MODEL)
    model = AutoModel.from_pretrained(cfg.bi_model or DEFAULT_MODEL).to(device)
    model.train()
    if cfg.bi_grad_ckpt:
        model.gradient_checkpointing_enable()

    n = len(anchors)
    H = negatives.shape[1] if negatives is not None and negatives.size else 0
    steps = math.ceil(n / cfg.bi_batch) * cfg.bi_epochs
    opt = AdamW(model.parameters(), lr=cfg.bi_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.bi_lr, total_steps=max(steps, 1),
                                                pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(0)
    log(f"  bi-encoder: {n:,} anchors, {H} mined negatives each, batch {cfg.bi_batch}, "
        f"{cfg.bi_epochs} epoch(s), device={device.type}")

    step = 0
    for epoch in range(cfg.bi_epochs):
        order = rng.permutation(n)
        run, seen = 0.0, 0
        for s in range(0, n, cfg.bi_batch):
            idx = order[s:s + cfg.bi_batch]
            B = len(idx)
            docs = list(positives[idx])
            if H:
                docs += [t for row in negatives[idx] for t in row if t]
            with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                qa = encode_batch(tok, model, anchors[idx], device, cfg.bi_max_len)
                dd = encode_batch(tok, model, docs, device, cfg.bi_max_len)
                # Row i of the batch must pick document i out of every positive and every
                # mined negative in the batch: B true docs followed by B*H hard ones.
                logits = (qa @ dd.T).float() / cfg.bi_temp
                target = torch.arange(B, device=device)
                loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step + 1 < steps:
                sched.step()
            step += 1
            run += float(loss) * B
            seen += B
            if step % cfg.bi_log_every == 0:
                log(f"    epoch {epoch} step {step}/{steps} loss={run / max(seen, 1):.4f}")
                run, seen = 0.0, 0

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"  bi-encoder saved to {out_dir}")
