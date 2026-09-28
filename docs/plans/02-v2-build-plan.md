> **SUPERSEDED for sequencing — do not build from this.** Written before the v2 hold-out
> logs existed, so its phase order assumes blocking is the bottleneck; the logs show the
> classifier is. Follow [../architecture.md](../architecture.md) instead.
> Still useful as reference: §0.5 (model licences, verified) and §0.6 (the out-of-memory
> diagnosis — the joblib sparse-broadcast bug and the measured threads fix).

# v2 Build Plan — from 0.90 to 0.99+

Execution plan. Sequenced for maximum final score, with a decision gate after every phase so
that effort is never spent on a stage that the measurements say is not the bottleneck.

Companion document: [../data-analysis.md](../data-analysis.md) holds the measured evidence
that every choice below rests on. This file is the *what to do, in what order*.

---

## 0. Assumptions, and what would change the plan

### Hard rules (from `student_resource/README.md`, §Constraints and §Fair Play)

1. **The final model must be MIT or Apache 2.0 licensed and at most 8 billion parameters.**
   This disqualifies several otherwise-attractive models — see §0.5.
2. **No external data lookup of any kind.** No entity APIs, no business registries, no
   **geocoding APIs for address normalization**, no internet data augmentation. Violation is
   immediate disqualification. Pretrained open-weight models are permitted — the 8B/licence
   rule presupposes them — but every alias and gazetteer table in Phase 1 must be **mined
   from the provided data only**. This is why the France region↔department table has to come
   from pseudo-labelled test clusters rather than a lookup.

### Compute budget (verified)

All work runs on one local machine: **16 GB RAM, RTX 4060 (8 GB VRAM), Windows**. There is no
Kaggle quota and no session cap, so GPU time is effectively unlimited — but it is *serial*, on
one consumer card, and the machine is unusable while a long pass runs.

**The binding constraints on model choice are 8 GB of VRAM and wall-clock time, not the 8B
parameter rule.** An 8B model is legal but will not load in 8 GB, let alone score tens of
millions of pairs. Every model choice in §0.5 is sized against this budget.

Rough orders of magnitude, **all of which must be benchmarked on the actual card before any
full run is launched** — they are extrapolations, not measurements:

| pass | volume | rough estimate |
|---|---|---|
| `multilingual-e5-small` embedding | 11.7M records, seq 64 | ~1–2 h |
| `xlm-roberta-base` cross-encoder inference | ~17M pairs, seq 96 | **~6–13 h** |
| `Qwen3-Reranker-0.6B` inference | ~1–2M pairs (banded) | ~2–4 h |
| `xlm-roberta-base` fine-tuning | a few M pairs, a few epochs | ~4–8 h |

The cross-encoder pass is an overnight job, not a coffee break. Halving the sequence length
from 96 to 64 roughly halves it, and is the first lever to pull if the measured rate is worse
than hoped.

### Assumptions

| Assumption | If false |
|---|---|
| There is enough calendar time for all seven phases | Cut from the bottom: phases 6 and 7 are the droppable ones. Phases 1–5 are not. |
| The RTX 4060 is usable for multi-hour unattended runs | Phase 7 is cut entirely. Phases 1–6 are CPU-only and still reach an estimated 0.96–0.98. |
| Leaderboard submissions are limited per day | Budget them as in §9 — never spend one on an unvalidated change. |
| The machine has 16 GB RAM | §0.6 keeps peak under 9 GB by sharding; if a stage still exceeds it, shrink the shard. |

Current state: v1 scored **0.90** on the portal. Existing pipeline is ~900 lines across
`src/` (`textnorm`, `retrieval`, `features`, `decision`, `pipeline`, `learn_aliases`,
`metric`, `eval_holdout`, `data`). It is small enough that v2 is a rebuild of stages, not a
rewrite of the repository.

**Standing rule for every phase:** nothing is generated inside `student_resource/`.
Work goes to `../../work`, outputs to `../../output`.

---

## 0.5 Model selection, with the evidence

Three separate model decisions, each with a different answer. They are not interchangeable,
and the biggest mistake available here is to spend effort swapping the tabular model when the
missing capability is representational.

### Decision 1 — the tabular pair scorer: keep LightGBM

An honest correction to an earlier claim in this project: "GBDTs always beat deep learning on
tabular data" is **no longer true as a general statement in 2026**. On the TabArena benchmark,
tabular foundation models (TabPFN-3, TabPFN-2.5/2.6, TabICLv2, TabFM) now lead single-model
Elo by a wide margin, and TabPFN-3 reportedly beats 8-hour-tuned GBDT baselines on datasets up
to 1M rows and 200 features. Notably, once hyperparameter ensembling is allowed, CatBoost
moves back ahead of the neural approaches.

It nevertheless remains the right choice **here**, for two reasons specific to this
competition rather than for the general reason:

1. **Licence.** TabPFN-2 weights ship under the *Prior Labs License* (Apache 2.0 **plus** an
   attribution requirement), which is not plain Apache 2.0. TabPFN-2.5, 2.6 and 3 are
   explicitly **non-commercial**. The competition demands MIT or Apache 2.0. Ruled out.
2. **Scale is inverted.** Tabular foundation models are in-context learners whose sweet spot
   is *small* training sets, and TabPFN degrades past ~10k rows. This problem is the opposite:
   ~54 features but **hundreds of millions of candidate pairs at inference**. LightGBM scores
   those in minutes on CPU; an in-context model cannot.

Worth testing once Phase 4 exists: **CatBoost for the stacker specifically**, because ordered
boosting and native categorical handling tend to give better-calibrated probabilities before
isotonic — and Phase 4's expected-F0.5 rule depends on absolute probability quality. Expected
gain is small (+0.1–0.3), so it is a late-stage refinement, not a priority.

### Decision 2 — the retrieval encoder: small and multilingual, sized to the quota

11.7M records (1.73M S1 + 4.89M S2 + 5.08M S3) must be embedded. Candidates, all
licence-compliant:

| model | params | licence | est. time for 11.7M records |
|---|---|---|---|
| `multilingual-e5-small` | 118M | MIT | ~1–2 h |
| `bge-m3` | 568M | MIT | ~5–7 h |
| `Qwen3-Embedding-0.6B` | 600M | Apache 2.0 | ~5–8 h |
| `Qwen3-Embedding-8B` | 8B | Apache 2.0 | infeasible |

`Qwen3-Embedding-8B` ranked first on the MTEB multilingual leaderboard, and is legal at
exactly the 8B limit — but embedding 11.7M records with it would consume the entire weekly
quota several times over. **Start with `multilingual-e5-small`** and only move up if Phase 2
recall misses its gate; the records are short (a name plus an address, well under 64 tokens),
which is the regime where small encoders lose least. `bge-m3` is a possible upgrade — it produces
dense, sparse and ColBERT-style multi-vector representations from one model, which fits the
multi-view design directly — **but check the embedding store before reaching for it.** At
11.7M records:

| encoder | dims | fp16 store |
|---|---|---|
| `multilingual-e5-small` | 384 | 9.0 GB |
| `multilingual-e5-small`, Matryoshka-truncated | 256 | **6.0 GB** |
| `bge-m3` | 1024 | **24.0 GB** |

On a 16 GB machine `bge-m3` at full width is not viable, and even e5-small at 384 dims must be
memory-mapped rather than held in RAM (§0.6). Truncating to 256 dims is the sane default here;
it is a capacity decision, not a tuning knob.

The literature is explicit that for bi-encoders, **embedding-pretrained variants give stronger
initialization** than general LLM backbones — so initialize from an embedding model, never
from a raw causal LM.

### Decision 3 — the reranker: a two-tier design, and this is where France is won

The most directly relevant finding in the current literature is a 2026 controlled factorial
study over bi-encoder / cross-encoder / generative matcher architectures (1,215 fine-tuning
runs across the Qwen3 family and nine datasets). Its conclusions map onto this problem
unusually well:

- **Cross-encoders consistently beat bi-encoders**, because they jointly encode the pair
  rather than embedding each record independently. Larger models narrow but do not close this.
- **Generative matchers beat cross-encoders mainly under distribution shift** — including
  unseen schema differences and cross-dataset transfer — **not by default**.
- **Larger models lean more on shortcut learning**, so bigger is not reliably better.

France is precisely a distribution shift: 15% of test S1, zero training signal, and the only
country where S1 itself is dirty. So the architecture that the literature says wins under
shift is the one to apply *there*, and the cheaper architecture everywhere else:

| tier | model | scope | est. cost |
|---|---|---|---|
| **A. bulk cross-encoder** | fine-tuned `xlm-roberta-base` (278M, MIT) | all ~17M pruned pairs | ~2–3 h |
| **B. generative reranker** | `Qwen3-Reranker-0.6B` (Apache 2.0) | France + the ambiguous band (~1–2M pairs) | ~1–2 h |

Tier B scores by reading the yes/no token logits of a generative model, which is exactly the
generative-matcher architecture the study found to transfer best. Restricting it to the band
where tier A is uncertain keeps it inside the quota while spending the expensive model on the
pairs that actually decide the score. `Qwen3-Reranker-4B` is the upgrade path if quota allows,
but the shortcut-learning finding argues against assuming it helps — measure it.

We hold **7.6M labelled pairs**, which is an unusually large in-domain training set. That
strongly favours *fine-tuning a small model* over prompting a large one, and is an additional
argument against reaching for the 8B tier.

Both tier outputs enter the LightGBM stacker as features. Neither replaces it.

---

## 0.6 Execution on 16 GB RAM + 8 GB VRAM — why it crashes and how it stops

v1 dies partway through the data on a 16 GB Windows laptop. **This is a code defect, not a
hardware limit.** It reproduces on any OS that uses `spawn` for multiprocessing, which is both
Windows and Python 3.14 on macOS — neither gets copy-on-write, so every worker receives a full
private copy of whatever it is sent.

### Measured cause

TF-IDF index sizes, extrapolated from a real 80k-document India sample to full India S2
(2,017,799 rows):

| view | full-slice size |
|---|---|
| address word 1–2gram | 0.43 GB |
| name `char_wb` 3–4gram | **2.34 GB** |
| address `char_wb` 4gram | 1.08 GB |
| **D total** | **3.85 GB** |

Four defects compound on top of that number:

1. **`topk_sparse` broadcasts the index to every worker.** `Parallel(n_jobs=-1)` passes `Q` and
   `DT` into each task. joblib memory-maps large NumPy arrays but **not scipy sparse matrices**,
   which are pickled whole. With the README's `--jobs 16`, that is 16 private copies of roughly
   5.5 GB.
2. **`DT = D.T.tocsr()` duplicates the index.** `.T` alone is a free view; `.tocsr()`
   materializes a second copy, and `D` is never released.
3. **`build_pairs` accumulates every country in a list, then `pd.concat`s**, doubling peak at
   the worst possible moment.
4. **`chunk=500`** creates ~1,700 tasks per country, each re-pickling those multi-GB matrices.

One (country, source) slice therefore needs D 3.85 GB + DT 3.85 GB + Q ~1.7 GB ≈ **9.4 GB
before any parallelism**, then multiplies by worker count. Raw data is not implicated at all:
pandas 3.0.6 uses Arrow-backed strings, so all three sources together measure ~1.6 GB.

### Budget

Windows itself takes 3–4 GB. **Target peak for the pipeline: ≤ 9 GB**, so the pagefile is never
touched — once Windows starts paging a sparse workload, it thrashes rather than slows.

### Fixes, in dependency order

1. **Threads, not processes, for sparse top-k.** scipy's sparse matmul releases the GIL, so
   `prefer="threads"` shares one copy of the index across workers. Measured on real data:
   **3.24× faster than sequential** *and* flat memory. This single change removes the dominant
   term. Raise `chunk` to ~2,000 at the same time to cut task overhead.
2. **`del D` immediately after building `DT`.** Frees 3.85 GB per slice.
3. **Document-side sharding with a frozen vocabulary.** Split each (country, source) pool into
   ~500k-row shards, compute top-k against each shard, and merge the per-shard top-k lists.
   Merging top-k across a partitioned corpus is **exact**, not an approximation. Memory becomes
   O(shard), not O(corpus): ~580 MB for the heaviest view instead of 2.34 GB.
   The vocabulary and IDF must be **fixed across shards** or scores are not comparable — fit the
   vectorizer once on a sample and `transform` each shard, or use a `HashingVectorizer` with a
   fixed feature count and a precomputed IDF, which keeps no vocabulary in memory at all.
4. **Query-side chunking.** Process S1 in blocks of ~50k rows against each shard.
5. **Never concatenate across shards in memory.** Each (country, source, S1-block) writes its
   own Parquet part; downstream stages read parts lazily. This deletes defect 3 outright and is
   what makes the stage resumable.
6. **Downsample negatives before training.** ~50M pairs × 54 float32 features is 10.8 GB and
   will not fit. Keep every positive, sample negatives by retrieval rank, and train on 5–10M
   pairs — ample for a GBDT. Build the `lgb.Dataset` with `max_bin=63` and `free_raw_data=True`:
   LightGBM bins to uint8, so 10M × 54 features costs ~540 MB once the raw frame is released.
   Calibrate isotonically on an **unsampled** validation split so the sampling rate is corrected.
7. **Memory-map the embeddings.** 11.7M records at 384 dims fp16 is 9.0 GB on disk — never in
   RAM. Write once, `np.memmap` thereafter. Matryoshka truncation to 256 dims brings it to
   6.0 GB and is the better default on this machine.

Applied together, peak drops from ~9.4 GB × workers to roughly **1.5–2 GB regardless of corpus
size**, which also removes the need to subsample the pools during validation — the thing that
produced the misleading 0.99.

### Where each stage runs

The 8 GB VRAM changes the split materially: most neural work fits locally, which preserves the
machine for every stage, so the real currency is wall-clock hours rather than a quota.

| stage | where | note |
|---|---|---|
| preprocess, retrieval, features, GBDT, decision | **local CPU** | after the fixes above; sharded and resumable |
| `multilingual-e5-small` embedding of 11.7M records | **local GPU** | fp16, batch 512, seq 64 — fits 8 GB comfortably |
| `xlm-roberta-base` cross-encoder fine-tune | **local GPU** | fp16, batch 16–32 at seq 96 with gradient accumulation |
| `Qwen3-Reranker-0.6B` inference | **local GPU** | ~1.2 GB fp16 weights; LoRA fine-tuning also fits |
| full-scale cross-encoder inference over ~17M pairs | **local GPU, overnight** | the single longest pass in the project; must checkpoint (see below) |
| anything 4B or larger | **nowhere** | 8 GB VRAM cannot hold it, and §0.5 already argues against it |

### Consequences of having no Kaggle fallback

1. **Checkpoint every long GPU pass.** With no session cap the runs get *longer*, not shorter,
   and a Windows update, a thermal shutdown or a stray reboot at hour nine costs the whole
   pass. Write scored pairs to Parquet in shards as they are produced and skip completed
   shards on restart. This is architecture item 4, and on this setup it is mandatory.
2. **8 GB VRAM is tight for fine-tuning, not just inference.** `xlm-roberta-base` in fp16 with
   AdamW needs roughly 0.6 GB weights + 0.6 GB gradients + 3.3 GB optimizer state, before
   activations — around 6 GB at batch 16 / seq 96. It fits, but only just. Gradient
   checkpointing and 8-bit Adam each buy headroom; note that `bitsandbytes` and
   `flash-attention` are both historically awkward on Windows, so verify they install before
   designing around them.
3. **Thermal throttling is a real variable on a laptop card.** Benchmark sustained throughput
   over ten minutes, not thirty seconds, or every estimate above will be optimistic.
4. **Nothing is parallel any more.** On Kaggle a long inference pass could overlap with local
   CPU work. Here the GPU pass and the CPU pipeline contend for the same box, so schedule the
   overnight GPU passes and the daytime CPU iteration explicitly.

---

## Phase 0 — Validation harness and diagnosis

**This is the only phase with no optional parts.** Everything downstream is unmeasurable
without it, and the 9-point gap between the 0.9918 dry-run score and the 0.90 portal score
is entirely a validation failure.

### Build

1. **Entity-aware splitter.** Split by S1 entity and hold out whole clusters. A record from a
   held-out cluster must never appear in the training pool — under the one-owner constraint,
   its presence leaks the answer.
2. **Full-pool scorer.** Held-out S1 entities are scored against the **complete** S2/S3 pool
   for their country, never a sampled one. This is the single most important property of the
   harness: lookalike density in a 5,000-entity sample is wrong by roughly three orders of
   magnitude, which is exactly why the dry run read 0.99.
3. **Breakdown reporting.** Every evaluation emits, as a matter of course: F0.5 overall; F0.5
   per country; F0.5 per true-cluster-size bucket (0, 1, 2, 3, 4+); F0.5 on singletons alone;
   precision and recall separately; candidate recall overall and per retrieval view;
   reduction ratio; wall-clock and peak memory.
4. **Leave-one-country-out mode.** Train on US, validate on India, and the reverse. This is
   the only available proxy for France.
5. **Frozen validation slices**, written once and reused so that every later number is
   comparable: `VAL-200K` (200k S1, mixed countries, full pools) as the standard gate, and
   `VAL-LOCO` for the country-transfer test.

### Diagnose

Run v1 unchanged against the harness and resolve the three competing hypotheses:

| | Hypothesis | Confirmed if |
|---|---|---|
| H1 | France/unseen-country generalization | LOCO score is far below the same-country score |
| H2 | Blocking recall capped near 85% | Candidate recall on VAL-200K is below ~97% |
| H3 | Decision rule mis-tuned | Singleton-only F0.5 is poor, or precision ≫ recall, or the reverse |

### Gate

Do not start Phase 1 until VAL-200K reproduces something close to 0.90 and the three
breakdowns above exist. **If the harness says 0.99, the harness is wrong** — most likely the
pools are still being sampled. Fix it before proceeding.

### Records to keep

A results table, appended to on every experiment thereafter: experiment ID, what changed,
VAL-200K F0.5, per-country F0.5, candidate recall, runtime. This table is what prevents
guessing later.

---

## Phase 1 — Normalization overhaul

Highest return per hour of work. Measured: US address token Jaccard on **true** pairs is only
16.3%, while house-number overlap is 87.7%. That gap is not weak signal — it is reordering
and abbreviation that normalization is failing to remove.

### Build

1. **Address component parser.** Treat comma-separated components as an **unordered set**,
   because they are demonstrably shuffled between sources. Extract `house_number`, `unit`,
   `street_tokens`, `locality`, `admin_area`, `postal`, `landmark`.
2. **Number handling.** Keep a multiset of numbers per address, reducing `3280 1/2`,
   `H.no 348`, `Door No 348` and `(41)` to comparable values. Numbers are compared later by
   digit edit distance, never equality — the noise (2657→2658, 247→47) is real and frequent.
3. **Unicode layer.** NFKC, casefold, strip zero-width and control characters, fold
   diacritics — while retaining a raw-script view for the encoder in Phase 7.
4. **Street-type canonicalization across all three languages.** St/Rd/Ave/Blvd/Dr/Ln for US;
   Rue/Av/Bd/Allée/Impasse/Route/Chemin for France, including the abbreviated forms actually
   present in the data (`R.`, `R`, `AV`, `ALL`, `IMP`, `RTE`).
5. **Legal-suffix canonicalization.** Grouped variants, kept as a **separate categorical
   feature** and stripped from the core name. Different suffixes can indicate different
   companies, so the information must be preserved, not discarded. Includes the French set
   (SARL, SAS, SASU, EURL, SCI, SNC, S.A., S.A.S) and bracketed forms.
6. **Name repair.** Leetspeak inside alphabetic tokens; junk prefixes (`>>`, `<<`, `...`,
   `((INC))`); store numbers (`#34301`) moved to a side feature; URL segmentation
   (`akshayagases.com`); initialism generation so `WM` can reach `White Management`; a
   sorted-token view for word-order invariance.
7. **Alias mining, data-driven only, no external gazetteers.**
   - US/India: mine state and street-type aliases from train ground-truth co-occurrence.
   - **France: mine from pseudo-labelled test clusters.** Group France records by house
     number plus street tokens, then read off which `admin_area` tokens co-occur. This is
     how `Nord ⊂ Hauts-de-France` and `Gironde ⊂ Nouvelle-Aquitaine` get learned without
     labels. The `admin_area` field must be a **soft alias feature, never a hard key or an
     equality test** — it is the exact analogue of the Williamsville/Buffalo problem.

### Verify

- Re-measure true-pair address Jaccard per country on the sample. **Target: US from 16.3% to
  above 45%.** If it does not move, the parser is not doing its job.
- Re-measure the blocking-key recall table. Every key should improve.
- Eyeball 50 normalized France records specifically, since France has no labels to check against.

### Gate

VAL-200K improves, and US address Jaccard clears 45%. Expected gain **+4 to +8**.

---

## Phase 2 — Retrieval

Measured: hard-key blocking tops out at 83–86% union recall. That is a hard ceiling sitting
below the target, and no downstream model can recover from it.

### Build

Four views, top-30 each, merged by reciprocal-rank fusion, capped near 60 candidates per S1
per source:

| view | representation | measured ceiling |
|---|---|---|
| A. address sparse | IDF-weighted word + char n-gram TF-IDF, house number as its own token | 95–96% |
| B. name char 3-gram sparse | TF-IDF | 98% US / 80% India |
| C. name dense | `multilingual-e5-small` bi-encoder (§0.5) | covers the India/Devanagari gap in view B |
| D. joint dense | same encoder over `name [SEP] address` | mixed evidence |

Plus **reverse retrieval**: every S2/S3 record retrieves its own top-10 candidate S1 owners.
This is not an optimization — it is what supplies the competition features in Phase 3 and the
"no owner" evidence needed for the 26% of records that are decoys.

Sharding: 1.73M S1 × ~120 candidates ≈ 200M pairs. Shard by country, store int32 pairs in
Parquet, never materialize the full pair table in memory.

### Verify

Candidate recall per view and for the union, per country, on VAL-200K, alongside the
reduction ratio. Views A and B are complementary across countries — confirm that the union
genuinely exceeds each view alone rather than one view dominating.

### Gate

**Union candidate recall ≥ 99.3% at ≤60 candidates per S1.** Below 99%, do not move on —
recall lost here is unrecoverable and caps the final score directly. Expected gain **+3 to +6**.

---

## Phase 3 — Pair scoring

### Build

1. **GBDT pruner** over cheap features, keeping the top 8–12 candidates per S1: retrieval
   scores and ranks per view, RRF score, rapidfuzz token_set/token_sort/Jaro-Winkler on name
   and address, digit overlap and digit edit distance, suffix-class agreement, length ratios,
   missing-field flags, script class.
2. **LightGBM stacker** over the pruner outputs plus **group-context features**, which are
   the ones that actually separate a good pipeline from an average one:
   - margin between this S1 and the runner-up S1 for the same record;
   - address exclusivity — how many distinct S1 share this address;
   - the record's best score to *any* S1 (uniformly low ⇒ it is a decoy);
   - cluster-mate agreement — how many of the record's likely mates also point at this S1;
   - candidate-count and cluster-size priors (shapes 1-1, 1-2, 2-1 dominate at 12.2/11.4/10.1%).
3. **Isotonic calibration** on the held-out split. Calibration is a prerequisite for Phase 4,
   not a nicety: expected-F0.5 selection is meaningless on uncalibrated scores.

Model choice for both stages is LightGBM, for the licence and inference-scale reasons set out
in §0.5. CatBoost is a worthwhile late experiment for the stacker alone. Tabular foundation
models are ruled out on licence grounds and do not fit the scale.

**One genuine upgrade inside LightGBM:** the current `LGBMClassifier` treats each pair
independently under binary logloss, but the data has natural groups — candidates per
(S1, source) — and the metric is per-entity. A LambdaRank objective grouped by S1 optimizes
ordering within an entity, which is closer to what is scored. The catch is that ranking
objectives produce good orderings but poor *absolute* probabilities, and Phase 4 needs
calibrated absolutes. So do not swap the objective: **add the ranker's score as a feature**
and keep logloss as the probability source.

### Hard negatives

Both directions of the confusion must be in the training set, because the data contains both:
different businesses sharing a street, and the same business under an unrelated name. Mine
(i) same-street different-entity pairs, (ii) same-name different-city pairs, (iii) the
top-ranked false candidates from Phase 2.

### Gate

VAL-200K improves and the calibration curve is close to diagonal.

---

## Phase 4 — Decision layer

Cheap, no GPU, and licensed directly by the measurements. This is the best
effort-to-points ratio in the plan after Phase 1.

### Build

1. **Global assignment with an explicit "nobody" option.** Each S2/S3 record chooses at most
   one S1 owner: softmax over its candidate owners plus a learned no-owner logit, kept only
   when the leader wins by a margin, ties resolved by mutual best match. Justified by the
   measured 0.0000% ownership collision across 7,638,365 matched IDs, and required by the
   ~26% decoy rate.
2. **Per-S1 expected-F0.5 output selection**, replacing the global threshold. Sort the
   calibrated probabilities and, for each cutoff k = 0…K, estimate the expected F0.5 by
   Poisson-binomial DP (exact, preferred) or Monte-Carlo. Choose the best k. The k = 0 case is
   the empty list, whose expected value is P(no true match) — this is how the 5.58% singletons
   get scored correctly, and each one is worth a full 1.0 or a full 0.0.
3. Keep whichever of expected-F0.5 and the best tuned global threshold wins on VAL-200K.

> `work_dry/config.json` currently reads `"mode": "threshold"`, `"thr": 0.7` — tuned on a
> 5,000-entity dry run. The expected-F0.5 path exists in `decision.py` but is not active.

### Gate

Singleton-only F0.5 rises materially, and overall precision improves without a
disproportionate recall loss. Expected gain **+2 to +4**.

---

## Phase 5 — Full-scale run and a real submission

Before any further modelling, produce a complete, validated submission from phases 1–4 and
spend one leaderboard slot on it. This converts the plan's estimates into a measured number
and confirms the full-scale path works end to end.

Checklist: exactly one row per test S1 (1,732,544); empty lists for predicted singletons; only
S2/S3 IDs that exist in test; matches a subset of candidates; both TSVs written; validator run
**with `--check-ids`**; outputs outside `student_resource/`.

The gap between VAL-200K and this portal score is itself the most valuable diagnostic in the
whole project — it tells you how much France is costing, which no local measurement can.

---

## Phase 6 — Group inference

Recovers the residue that pair-level scoring provably cannot reach: the **1.2–1.7% of true
pairs where both channels are weak**, and the 3–5% of S2/S3 records with an empty address.
Pairwise methods have a measured ceiling around 98.5%; this phase is how you get past it.

### Build

1. Cluster S2/S3 records within a country that share a near-identical address and a
   compatible name. Noisy copies mirror each other, so these groups are tight.
2. **Address borrowing** — an address-less record inherits its group's address for scoring.
3. **Group evidence** — a record whose own name is a random string is scored using its
   group-mates' name evidence.
4. Guard against co-located businesses: merge only when name *and* address agree, and cap
   group size. Address exclusivity from Phase 3 is the safety check.

Expected gain **+1 to +2**.

---

## Phase 7 — Two-tier neural reranking

The main quality driver on the genuinely ambiguous band, and the most expensive item here.
GPU-only, run locally overnight on the RTX 4060. Model rationale and evidence are in §0.5.

### Tier A — bulk cross-encoder over all surviving pairs

- Backbone `xlm-roberta-base` (278M, MIT); `mdeberta-v3-base` is the alternative.
- Field-tagged input: `[NAME] … [ADDR] … [SEP] [NAME] … [ADDR] … [META] country, suffix,
  source`, max length 96–128 tokens.
- Train on Phase 3 survivors with the Phase 3 hard negatives. Both directions of the
  confusion must be represented, since the data contains both: same address with different
  businesses, and the same business under an unrelated name.
- Roughly **6–13 h for ~17M pairs** on an RTX 4060 (estimate — benchmark first). This is an
  overnight job and must checkpoint per shard. Dropping the sequence length from 96 to 64
  roughly halves it.

### Tier B — generative reranker on the pairs that decide the score

- `Qwen3-Reranker-0.6B` (Apache 2.0), applied to **France plus the ambiguous band** where
  tier A is uncertain (roughly 0.2–0.8), about 1–2M pairs, ~2–4 h on the 4060.
- Rationale: the 2026 factorial study finds generative matchers beat cross-encoders
  *specifically under distribution shift*, which is exactly what France is. Spending the
  expensive architecture only where it has a measured edge keeps the whole phase inside the
  30 h/week quota.
- `Qwen3-Reranker-4B` is **out of reach** on 8 GB VRAM regardless of merit — and the same study
  finds larger models lean harder on shortcut learning, so this is a smaller loss than it looks.

### Also worth doing

Fine-tune the Phase 2 bi-encoder contrastively (InfoNCE, in-batch negatives) on the same hard
negatives. This lifts retrieval and scoring together and is cheap relative to its effect.

Both tiers feed the Phase 3 LightGBM stacker as additional features. Neither replaces it.

### Measure before committing

Benchmark sustained throughput over ~10 minutes and extrapolate before launching a full run.
Every hour estimate in this document is an extrapolation from model size, not a measurement on
this card. Expected gain **+2 to +4**.

---

## 8. Expected trajectory

| after | expected VAL-200K | main driver |
|---|---|---|
| Phase 0 | ~0.90 measured honestly | — |
| Phase 1 | 0.94–0.96 | normalization, France |
| Phase 2 | 0.96–0.97 | recall ceiling lifted to 99.3% |
| Phase 3–4 | 0.97–0.98 | assignment, expected-F0.5, singletons |
| Phase 6 | 0.98–0.985 | group inference |
| Phase 7 | 0.99+ | cross-encoder on the ambiguous band |

These are estimates built from the measured ceilings in `data_analysis_v2.md`, not
guarantees. The per-phase gates are what keep them honest.

---

## 9. Submission budget

1. One submission at the end of Phase 5, to anchor the estimates against reality.
2. One after Phase 6.
3. One after Phase 7.
4. Remaining slots on decision-layer variants only — those are cheap to generate and are the
   most likely source of a final fraction of a point.

Never spend a submission on a change that has not first improved VAL-200K. The portal gives a
single aggregate number with no breakdown; it cannot tell you *why* anything moved, so it is
a confirmation instrument, not a development one.

---

## 10. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| France alias mining produces wrong region↔department pairs | Keep it a soft feature; verify by hand on 50 clusters; if it looks unreliable, drop `admin_area` from France scoring entirely rather than trusting a bad table |
| Full-scale run exhausts 16 GB locally | Country- and document-sharded processing throughout (§0.6); shrink the shard before changing anything else |
| A multi-hour GPU pass dies partway | Per-shard checkpointing on every long pass; there is no second machine to fall back to |
| Neural reranking takes too many wall-clock hours | Tier B is already band-restricted; cut sequence length to 64, then drop to tier A only, then to the stacker alone, which already reaches an estimated 0.98 |
| The pipeline OOMs again on the 16 GB laptop | §0.6 sets a ≤9 GB peak budget; if a stage exceeds it, shrink the document shard before touching anything else — shard size is the only free variable that does not change results |
| A chosen model turns out not to be MIT/Apache 2.0 | Verify the *weights* licence, not just the code licence, before training — TabPFN is the cautionary case, where Apache-2.0 code ships with non-commercial or attribution-encumbered weights |
| Group inference merges co-located businesses and costs precision | Require name *and* address agreement, cap group size, gate on VAL-200K precision before keeping it |
| Time runs short | Phases 1, 2 and 4 are the non-negotiable core; 6 and 7 are the droppable tail |

---

## 11. Open questions to resolve early

- Submission deadline and submissions-per-day limit — these set how much of the tail is reachable.
- Measured throughput on the RTX 4060 for each of the four GPU passes. Every hour figure in
  this plan is extrapolated from model size and must be replaced with a real number before the
  schedule is trusted.
- Whether renting a GPU for the final inference passes is acceptable. A single cloud A100 day
  would collapse the 6–13 h cross-encoder pass and bring `Qwen3-Reranker-4B` into range; the
  plan does not assume it, but it is the cheapest available speed-up if the deadline tightens.
- Measured throughput for each chosen model, which is currently estimated (§0.5) and must be
  benchmarked before the first full run.

Resolved since the first draft: the model rules are MIT/Apache 2.0 and ≤8B parameters, no
external data lookup; all work runs on one local Windows machine with 16 GB RAM and an
RTX 4060 (8 GB VRAM), with no Kaggle involvement; and the v1 out-of-memory crash is diagnosed
in §0.6 as a code defect rather than a hardware limit.
