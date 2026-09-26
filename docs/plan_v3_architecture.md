# Plan v3: architecture for the highest possible score on a 16 GB RAM / RTX 4060 laptop

Assumptions: RTX 4060 laptop GPU = 8 GB VRAM, 16 GB system RAM, 1 TB SSD, Windows (WSL2 recommended).
Numbers marked (est.) are my estimates, not measurements. Numbers from your logs are marked (log).

---

## 0. Honest target and error budget

Where you stand (log): v1 realistic hold-out F0.5 = 0.9388; v2 held-out = 0.9562 with candidate recall 0.953 and an oracle ceiling of 0.990.

To reach 0.99 you have to remove two separate losses:

| Loss | Today (v2) | What must be true for 0.99 |
|---|---|---|
| Candidate ceiling (true match never shortlisted) | 1.0 point (ceiling 0.990 at recall 0.953) | pair recall >= 99.5% (ceiling ~0.998) |
| Classifier error inside the shortlist | ~3.4 points (0.990 -> 0.956) | precision and recall inside the shortlist both ~99%+ |
| India | recall only 0.86 to 0.94 depending on the view (log) | closes only with a multilingual retriever |
| France | unmeasured (zero-shot) | must behave like US/India |

Verdict: 0.99 is a stretch goal, not a promise. A realistic outcome of the full plan is **0.97 to 0.98**, and 0.99 is only possible if the error analysis in step 1 shows the remaining errors are fixable (not genuinely ambiguous records). Step 1 decides how far to push. Do it first.

---

## 1. Design rules forced by the hardware

1. **Never hold a big pair table in RAM.** Process one (country, source) shard at a time, store candidate tables as Parquet on the SSD, use int32 IDs and float16/float32, and prefer polars or Arrow over pandas. (Your India predict already produced 24M pairs (log), so the current pandas flow will not fit in 16 GB with the new features.)
2. **Embeddings live on disk** as float16 memmaps: 768 dims x 2 bytes = 1.5 KB per record, so 11.7M test records = about 18 GB. Fine for a 1 TB SSD.
3. **GPU does the heavy lifting in chunks.** Search by brute-force matrix multiply in chunks (torch, fp16) instead of FAISS-GPU, which has no Windows build. This is fast enough (see section 6).
4. **Use WSL2 (Ubuntu) with CUDA passthrough** for the GPU parts. Cap WSL memory in `.wslconfig` to about 12 GB so Windows keeps some.
5. **Everything checkpoints** every N steps and can resume. Laptops throttle, sleep and crash.
6. **Compute-heavy fallback:** if the laptop is too slow for a stage, that stage can run on Kaggle (T4 x2/P100) or on the AWS credits. Check whether the AWS Builder credits cover EC2 GPU instances before relying on them.

---

## 2. Model choice (licenses verified on Hugging Face model pages)

The rules require MIT or Apache-2.0 and at most 8B parameters.

| Role | Model | License | Size | Verdict |
|---|---|---|---|---|
| **Bi-encoder (retrieval), primary** | `intfloat/multilingual-e5-base` | MIT | ~278M | **Pick.** Multilingual (Hindi, French), small enough to fine-tune with large effective batch on 8 GB. |
| Bi-encoder challenger | `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | 596M | Stronger multilingual quality; needs LoRA and is about 2x slower. Try only if the primary leaves India recall < 99%. |
| Bi-encoder alternative | `BAAI/bge-m3` | MIT | ~568M | Very strong but heavy for 8 GB fine-tuning; inference only. |
| **Cross-encoder (pair judge)** | `xlm-roberta-base` (init from the fine-tuned e5-base weights if it validates better) | MIT | ~279M | **Pick.** Reads both records jointly. Comparable option: `microsoft/mdeberta-v3-base` (MIT); try only if time allows (slower, fp16 quirks). |
| Small/fast fallback | `paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | ~118M | 3x faster; use if throughput is the bottleneck. |
| Indic-specific idea | `google/muril-base-cased` | Apache-2.0 | ~237M | Trained on transliterated Indic text; a candidate cross-encoder for India only if the shared one is weak. |
| Rejected | LaBSE (Apache, 471M) | | | Trained on translation pairs, not entity strings; no advantage over e5. |
| Rejected | Large LLMs (7-8B) | | | Do not fit 8 GB VRAM for 30M+ pair inference in the time left. |

Why not off-the-shelf zero-shot: the dataset gives about 7.6M labelled matches with systematic noise (suffix moves, glued domains, native-script names, noisy house numbers). A model fine-tuned on those beats any generic checkpoint. Every model above is used only as a pretrained starting point.

---

## 3. Architecture: Retrieve -> Rank -> Resolve

```
raw TSV
  |
  v  [L0] Canonical parse + cache (Parquet shards per country/source)
  |
  v  [L1] RETRIEVAL: union of views, target pair recall >= 99.5%
  |       A address words (TF-IDF)          C name letter-chunks + phonetic skeleton
  |       B address letter-chunks           D fine-tuned dense bi-encoder (joint name+address)
  |       E reverse retrieval (S2/S3 -> S1)  F sibling expansion (2-hop, see L4)
  |
  v  [L2] PRUNE: LightGBM on lexical + embedding-cosine features, keep top ~8 per S1 per source
  |
  v  [L3] JUDGE: fine-tuned cross-encoder, only on the uncertain band (gated), then LightGBM stacker
  |
  v  [L4] RESOLVE: sibling graph features -> global one-owner assignment -> per-S1 expected-F0.5 -> output
```

### L0. Data layer
- Reuse your current normalisation (phonetic skeleton, French street vocabulary, name abbreviations) and add domain/handle splitting and a digit-tolerant house-number feature from my local branch.
- Write per-(country, source) Parquet shards. Keep `entity_id -> int32` maps.

### L1. Retrieval (the largest single gain; India first)
- Lexical views A/B/C exist (your logs). Dense view D is new.
- **Bi-encoder recipe:**
  - Input text `name | address` (plus a separate name-only view if the diagnostic shows it adds recall). No country token, so France works.
  - Loss: InfoNCE with in-batch negatives via a cached/gradient-cached loss (large effective batch on 8 GB), plus 2-3 mined hard negatives per anchor (same street, same name different city, top wrong candidates from the lexical views).
  - Data: S1 anchor vs each matched S2/S3 record. Sample about 1.5M pairs, fp16, sequence length 64, one epoch.
  - Time on the 4060: about 1 hour of training (est.), plus hard-negative mining (about 30 min, est.).
- **Search:** embed each shard, then brute-force chunked matmul on GPU with a running top-K merge, per country. Total compute across US, India and France is about 1.4e16 FLOPs, or roughly 25 to 40 minutes on the 4060 (est.).
- **Embedding the test set:** 11.7M records at about 4,000 to 6,000 records/s (est.) = 35 to 60 minutes.
- **Measure recall per view** with your `eval_diag` script on unseen S1. Keep only views that add recall. Aim for pair recall >= 99.5% at <= 40 candidates per S1 per source.

### L2. Prune
- LightGBM (GPU or CPU) on lexical features plus dense cosine, rank per view, and the record-level competition features you already have. Keep the top 8 per S1 per source. Target: still >= 99.4% recall after pruning (log shows 0.9531 -> 0.9525 with your current prune, so pruning itself is fine).

### L3. Cross-encoder judge (the main classifier upgrade)
- Input: `[S1 name | S1 address] </s> [candidate name | candidate address]`, max length 128, fp16.
- Training data: about 1.5M pairs. Positives = true matches. Negatives = pruned candidates that are wrong (hard by construction), including same-address different-business and same-name different-city cases, and DBA positives with a different name.
- Time: about 350 pairs/s (est.), so about 70 minutes per 1.5M pairs.
- **Gated inference:** run the cross-encoder only where the LightGBM probability is uncertain (about 0.03 to 0.97). This is likely 20 to 30% of pairs (est.), so about 6 to 8M pairs at about 3,000 pairs/s (est.) = 35 to 45 minutes.
- **Stacker:** LightGBM over cross-encoder logit, lexical features, and dense cosine. Calibrate with isotonic regression per country.

### L4. Resolve (global, where precision is won)
1. **Sibling graph:** link S2/S3 records with near-identical addresses and compatible names (cheap: address TF-IDF, top-5 within a country). For each candidate (S1, record), add features from siblings' probabilities for the same S1: max, mean, count. Records with a missing address inherit the group's address evidence. Compute out-of-fold for training.
2. **Sibling expansion in retrieval (view F):** if a record is confidently linked to an S1, its siblings become candidates for that S1 even if their own text is poor. This targets the last fractions of recall.
3. **Global assignment:** every S2/S3 record belongs to at most one S1 (verified: 0 violations in train). Solve per connected component with the Hungarian algorithm including a "nobody" option, maximizing sum of (probability - threshold), instead of greedy one-owner. Cap 11 matches per S1.
4. **Per-S1 selection:** expected-F0.5 maximization over calibrated probabilities, including the empty list (5.6% singletons are worth a full 1.0 each). Tune thresholds per country.

### France (zero-shot)
- No country features anywhere. Multilingual encoders handle French.
- **Pseudo-label loop:** run the pipeline on France test, keep only pairs with probability > 0.98 that are also mutual best matches, fine-tune bi-encoder and cross-encoder for one short epoch on those plus a replay sample of US/India, then rerun France. Your `work_v2` already attempted a version of this.
- **Sanity checks against train priors:** matches per S1 about 3.46, singletons about 5.6%, at least one match about 94.4%. If France deviates wildly, the thresholds are wrong.

---

## 4. Validation protocol
- Split by S1 entity only. Keep a never-touched hold-out (your `eval_holdout`, full S2/S3 pool as negatives). Report recall, ceiling and F0.5 for every stage, per country.
- Report: candidate recall, oracle ceiling, F0.5 by stage (lexical only -> +dense -> +cross-encoder -> +assignment). This table also goes into the methodology document.
- **Error analysis first (step 1):** bucket the misses of v2 into (a) not shortlisted, (b) shortlisted but scored low, (c) false merges; then by noise type, country, and source. This tells us whether 0.99 is even reachable.

---

## 5. Order of work and time (hackathon ends 27 Sep; cut points are explicit)

| Step | Work | Time (est.) | Gate |
|---|---|---|---|
| 1 | Error analysis of v2 on held-out; measure per-view recall | 1-2 h | Decide the target (0.97 vs 0.99) |
| 2 | Merge repos (my local fixes onto your remote); submit the best existing output as a safety net | 1 h | A valid file is uploaded |
| 3 | Fine-tune the bi-encoder (overnight) and mine hard negatives | 2-3 h GPU | India recall >= 0.97 on held-out |
| 4 | Embed test, GPU search, union with lexical views, retrain prune | 3-4 h | Pair recall >= 99% |
| 5 | Cross-encoder training (about 1.5M pairs) | 2 h GPU | Beats stacker-without-it by >= 0.5 point held-out |
| 6 | Gated inference, stacker, sibling features, assignment, expected-F0.5 | 3-4 h | Held-out improves each stage |
| 7 | France pseudo-label loop and sanity checks | 2 h | France stats match priors |
| 8 | Final predict, validate, documentation, zip | 3 h | PASS on validator |

Each stage is independently submittable, so stop at any gate that passes and submit. If time runs short, cut in this order: France loop, sibling expansion, cross-encoder gating tweaks. Never cut steps 1 to 4: retrieval recall is the ceiling for everything after.

---

## 6. Risks
| Risk | Mitigation |
|---|---|
| 16 GB RAM overflow in feature building | Shard by (country, source), polars/Arrow, float32/int32, spill to Parquet |
| GPU throughput lower than estimated | Measure on 50k records first; drop to MiniLM (118M) or shorten sequences; offload to Kaggle/AWS |
| Laptop thermal throttling or sleep | Power plugged in, sleep disabled, checkpoints every 15 minutes, resumable scripts |
| Cross-encoder gain small | Gate decides; the pipeline stays valid without it |
| Errors in 0.99 gap are irreducible (identical or truly ambiguous records) | Step 1 tells us early; stop pushing and polish France instead |
| Audit: model licenses | All chosen models are MIT or Apache-2.0 (checked); record names and licenses in the methodology document |
| No external data | Only pretrained weights plus provided data; log this in the documentation |
