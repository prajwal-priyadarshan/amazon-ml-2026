# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A retrieve → rank → resolve pipeline. Hybrid retrieval (five sparse lexical views plus a
fine-tuned multilingual bi-encoder) reaches **99.6% pair recall** on held-out entities. A
cascade of learned scorers (LightGBM pruner, fine-tuned XLM-R cross-encoder, LightGBM
stacker) ranks the candidates. The key final addition is a **cluster-aware refine model
trained only on entities no upstream model ever saw**. Matches are then chosen per Source-1
entity by maximising expected F0.5 under a one-owner-per-record constraint. Held-out macro
F0.5: **0.9860** (open world) / **0.9885** (closed world), see §5.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training files:

- **One owner per record.** No S2/S3 record ever matches two S1 entities (0 of 7.64M).
  **26% of train records have no owner at all** (decoys).
- **The test set is denser in decoys.** Test has 5.75 pool records per S1 against 4.68 in
  train, so roughly twice as many decoys per entity.
- **Cluster shape.** 5.6% of S1 entities are singletons. The mean is 3.46 matches per
  entity, and the maximum is 11.
- **Name noise:** legal-suffix moves, glued domains (`akshayagases`), leetspeak, typos,
  native-script names (Devanagari, Tamil, Gujarati, …), and rebrands to invented names
  ("Rizapyra", "X F/K/A Y", "d/b/a").
- **Address noise:** abbreviations, reordered components, house numbers off by a few,
  region vs department (France), and **about 4% of records with no address at all**.
- **Decoys look like copies.** Examples are a sibling business at the same address with
  another legal form ("Nantes Medico SARL" vs "Nantes Medical S.A."), or a near-copy with
  a shifted house number ("60 Waterford Way" vs "600-604 …").
- **France is test-only** (15% of test S1) and has zero labels.

### 2.2 Solution Strategy

**Approach Type:** hybrid. Blocking by sparse and dense retrieval, then a cascade of
classifiers, then collective (cluster-level) inference and global assignment.

**Core Innovations:**

1. **A fine-tuned multilingual bi-encoder** (InfoNCE on train matches with mined hard
   negatives). It lifted pair recall from 0.932 to 0.996.
2. **A cluster-aware refine layer.** Each candidate is scored against the *other* confident
   copies of the same entity in both sources (name, address, house-number and legal-form
   consensus), against name rarity in the whole S1 table and the record pool, and against
   the competition for the record across all S1 entities.
3. **Honest training data for the top layer.** The encoder, judge and stacker were all
   trained on one split, so their scores are over-confident on that split. The refine model
   is trained on three *other* splits (850k fresh entities, 3.9M pairs), with S1-grouped
   cross-fitting.

---

## 3. Candidate Generation (Blocking)

- **Blocking views:** five sparse views and one dense view, fused with reciprocal-rank
  fusion. Each view keeps its top-k per S1 per source. All views run per (country, source)
  shard, with country treated as an open set.
  - word TF-IDF on the segmented name;
  - address TF-IDF;
  - character n-grams on the name;
  - a consonant skeleton;
  - exact keys (sorted name tokens, glued-name segmentation key);
  - dense: exact GPU top-25 cosine from the fine-tuned `multilingual-e5-small`.
- **Pruning:** a LightGBM pruner keeps the top 8 per S1 per source. A sibling-expansion
  view then adds same-address group mates of strong candidates.
- **Candidate pairs:**
  - about 202M retrieval pairs on test;
  - **9.2M pairs after pruning.** This is the set every later model scores, and it is
    what `candidate_pairs.tsv` contains.
- **Protecting recall:** recall was measured per view and per country on held-out
  entities before each cut.
  - Union recall is 0.9958. The oracle F0.5 ceiling on the candidates is 0.9987.
  - Pruning loses 0.2% of true pairs.

---

## 4. Matching Model

**Features used:**

- **Name features:** rapidfuzz ratios (ratio, partial, token-sort, token-set,
  Jaro-Winkler), segmented-name token set, exact/skeleton key equality, legal-form
  equality and conflict, extra and dropped words, IDF-weighted token Jaccard, and an
  out-of-vocabulary (invented brand name) share. Alias markers are also flagged (F/K/A,
  d/b/a, trading as).
- **Address features:** token-set ratio, IDF-weighted Jaccard and the IDF mass of missing
  S1 tokens, house-number agreement, missing-address flags, and address rarity in the pool.
- **Neural features:** fine-tuned bi-encoder cosine and rank. The cross-encoder logit
  (XLM-R base, fine-tuned for two epochs) is run on the uncertain band.
- **Competition and cluster features:**
  - per-S1 rank, gap and sum of scores across both sources;
  - the record's best competing S1;
  - how many S1 entities in the whole S1 table share the name;
  - agreement with the entity's other confident copies (name, address, house number,
    legal form, consensus house number and legal form);
  - same-address sibling scores.

**Model types:**

1. LightGBM pruner.
2. Cross-encoder `FacebookAI/xlm-roberta-base` (MIT, 279M).
3. LightGBM stacker with per-country isotonic calibration.
4. LightGBM refine model (3 seed bags) trained on hold-out splits.
5. Bi-encoder `intfloat/multilingual-e5-small` (MIT, 118M).

All models are MIT-licensed and well under 8B parameters. No external data is used: every
vocabulary, IDF table and alias list is mined from the provided files.

**Threshold selection method:** there is no global threshold.
1. **One owner per record:** each S2/S3 record keeps only its best S1.
2. **Per-S1 list:** each S1 entity gets the prefix of its candidates (by probability) that
   maximises Monte-Carlo **expected F0.5**. The empty list is one of the options, which is
   what scores singletons correctly.
3. **Validation:** this rule was chosen over a tuned global threshold on held-out entities.

---

## 5. Results & Error Analysis

**Held-out evaluation.** 150,000 S1 entities never used by any model, scored against the
full 10.3M-record train pool. "Open world" counts records owned by non-held-out entities
as free decoys, which is harsher than test. "Closed world" drops them, which is milder.

| stage | macro F0.5 (open) | closed |
|---|---|---|
| tabular cascade, lexical retrieval only | 0.9480 | — |
| + fine-tuned bi-encoder retrieval + cross-encoder judge | 0.9776 | — |
| + refine layer (holdout-trained, cluster features) | 0.9822 | 0.9857 |
| + full-S1-table rarity, legal-form, IDF, stacker design matrix | 0.9843 | 0.9872 |
| + two extra clean splits (hold2/hold3), 2nd judge epoch, 3 bags | **0.9860** | **0.9885** |

Leave-one-country-out (train refine on US only, score India) costs about 0.0026. This is
the only available proxy for France.

**Common false positives (wrong merges):**
- sibling businesses at the same address with a different legal form or an extra word;
- near-copies with a shifted house number (generated decoys);
- name-only records whose name is shared by another S1 entity.

**Common false negatives (missed matches):**
- records with **no address** and a generic name; these are the largest group;
- rebrands to an invented name ("Rizapyra") at an address that differs slightly;
- large clusters (6+) where one copy is heavily corrupted.

---

## 6. Conclusion

Retrieval quality set the ceiling: the fine-tuned bi-encoder moved it from 0.974 to 0.999.
The biggest ranking gain came from two things:
- describing each candidate against the rest of its cluster instead of in isolation;
- training the top layer on entities that no upstream model had seen.

Validating on data the models were trained on over-states confidence. Evaluating in both
open and closed world brackets the test condition.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- **`src/v3/`**, the pipeline:
  - `prep` (normalisation, segmentation);
  - `lexical`, `dense` (retrieval);
  - `pairs` (features);
  - `gbdt`;
  - `crossenc`, `train_bi` (neural);
  - `sibling`;
  - `refine` (cluster-aware top layer);
  - `resolve` (one-owner + expected-F0.5);
  - `run.py` (stage CLI).
- **`tools/run_v3.ps1`:** the tabular stages.
- **`tools/run_v3_neural.ps1`:** bi-encoder fine-tune, embeddings, dense search,
  cross-encoder.
- **`tools/run_v3_refine.ps1`:** extra clean splits, second judge epoch, refine, and
  resolve into the output folder.

Run the three scripts in that order from `code/business_entity_resolution/`. Every stage is
resumable. Full runtime is about 16 h on a 16 GB RAM / RTX 4060 laptop.

### B. Additional Results

Miss taxonomy on held-out entities (refine layer): 96.6% of true pairs hit, 2.7% scored
too low, 0.4% not retrieved, 0.2% pruned away. By bucket, records without an address are
hit 49% of the time (36% before refine), against 98.8% for records with an address.
