"""L3 cross-encoder judge: one transformer reading both records jointly.

Why a cross-encoder and not a bigger bi-encoder: a bi-encoder has to compress each record
into a vector before it has seen the other one, so it cannot reason about *which* token
disagrees. A cross-encoder attends across the pair, which is what decides cases like
"same street, different business" and "same business, unrelated name". The controlled
comparisons in the literature put cross-encoders consistently above bi-encoders on entity
matching, and larger models narrow that gap without closing it.

Two properties of this problem shape the recipe:

* **Negatives are free and already hard.** The training pairs come from the pruner's
  survivors, so every negative is a pair that lexical and dense retrieval both thought was
  plausible. There is no need to mine anything separately.
* **Inference is gated.** Running a transformer over every surviving pair is the longest
  pass in the project. Running it only where the pruner is *uncertain* costs a fraction and
  changes almost nothing, because a pruner probability of 0.001 or 0.999 is already right
  nearly always. Pairs outside the band keep ``ce_run=0`` and the stacker learns to read
  ``ce_logit`` only when it is present.

The judge reads raw text for the same reason the bi-encoder does: folding Devanagari to
"raam maarkettinng" destroys the signal an mBERT-family model was pretrained on.
"""
import math
import os

import numpy as np
import pandas as pd

from .dense import torch_device
from .paths import log

DEFAULT_MODEL = "FacebookAI/xlm-roberta-base"


def pair_texts(df):
    name = df["business_name"].fillna("").astype(str)
    addr = df["business_address"].fillna("").astype(str)
    return (name + " | " + addr).to_numpy(dtype=object)


def build_texts(cand, s1, d):
    a = pair_texts(s1)[cand["s1_pos"].to_numpy()]
    b = pair_texts(d)[cand["d_pos"].to_numpy()]
    return a, b


def _load(model_name, device, num_labels=1, train=False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if train:
        model.float()  # a saved judge may be stored in fp16; AMP needs fp32 master weights
    model.to(device)
    model.train(train)
    return tok, model


def train(out_dir, texts_a, texts_b, y, cfg, init_from=None):
    """Fine-tune the judge with BCE on one logit. fp16 autocast, AdamW, linear decay."""
    import torch
    from torch.optim import AdamW

    device = torch_device()
    tok, model = _load(init_from or cfg.ce_model, device, num_labels=1, train=True)
    n = len(y)
    steps = math.ceil(n / cfg.ce_batch) * cfg.ce_epochs
    opt = AdamW(model.parameters(), lr=cfg.ce_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.ce_lr, total_steps=max(steps, 1),
                                                pct_start=0.06, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    log(f"  cross-encoder: {n:,} pairs, {cfg.ce_epochs} epoch(s), batch {cfg.ce_batch}, "
        f"max_len {cfg.ce_max_len}, device={device.type}")

    step = 0
    for epoch in range(cfg.ce_epochs):
        order = rng.permutation(n)
        run_loss, seen = 0.0, 0
        for s in range(0, n, cfg.ce_batch):
            idx = order[s:s + cfg.ce_batch]
            enc = tok(list(texts_a[idx]), list(texts_b[idx]), padding=True, truncation=True,
                      max_length=cfg.ce_max_len, return_tensors="pt").to(device)
            target = torch.tensor(y[idx], dtype=torch.float32, device=device)
            with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                logits = model(**enc).logits.squeeze(-1)
                loss = loss_fn(logits.float(), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step + 1 < steps:
                sched.step()
            step += 1
            run_loss += float(loss) * len(idx)
            seen += len(idx)
            if (step % cfg.ce_log_every) == 0:
                log(f"    epoch {epoch} step {step}/{steps} loss={run_loss / max(seen, 1):.4f}")
                run_loss, seen = 0.0, 0
    os.makedirs(out_dir, exist_ok=True)
    if device.type == "cuda":
        model.half()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"  cross-encoder saved to {out_dir}")


class Judge:
    """A loaded judge, so a full pass does not reload the weights per block or per chunk."""

    def __init__(self, model_dir, cfg):
        self.cfg = cfg
        self.device = torch_device()
        self.tok, self.model = _load(model_dir, self.device, num_labels=1, train=False)
        if self.device.type == "cuda":
            self.model.half()

    def score(self, texts_a, texts_b):
        """Logits for a set of pairs, batched shortest-first so padding waste stays low."""
        import torch

        cfg = self.cfg
        n = len(texts_a)
        out = np.zeros(n, np.float32)
        # Vectorised length sort: a Python list comprehension over tens of millions of pairs
        # costs hundreds of MB before the model has seen anything.
        lens = (pd.Series(texts_a).str.len().to_numpy(np.int32)
                + pd.Series(texts_b).str.len().to_numpy(np.int32))
        order = np.argsort(lens, kind="stable")
        del lens
        with torch.inference_mode():
            for s in range(0, n, cfg.ce_infer_batch):
                idx = order[s:s + cfg.ce_infer_batch]
                enc = self.tok(list(texts_a[idx]), list(texts_b[idx]), padding=True,
                               truncation=True, max_length=cfg.ce_max_len,
                               return_tensors="pt").to(self.device)
                out[idx] = self.model(**enc).logits.squeeze(-1).float().cpu().numpy()
                if (s // cfg.ce_infer_batch) % 500 == 0:
                    log(f"    ce {min(s + cfg.ce_infer_batch, n):,}/{n:,}")
        return out


def score(model_dir, texts_a, texts_b, cfg):
    return Judge(model_dir, cfg).score(texts_a, texts_b)


def gate(prob1, lo, hi):
    """Mask of pairs worth a transformer pass."""
    return (prob1 >= lo) & (prob1 <= hi)
