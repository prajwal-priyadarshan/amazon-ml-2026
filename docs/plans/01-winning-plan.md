> **SUPERSEDED — do not build from this.** Pre-measurement plan. Follow
> [../architecture.md](../architecture.md). Kept for history only.

# Amazon ML Challenge 2026: Winning Plan (Entity Resolution)

Status: plan only, nothing here has been run yet. Facts marked (verified) were checked on the train files.

---

## 0. What the data actually looks like (drives every design choice)

Verified by sampling real train clusters:

| Finding | Evidence | Consequence |
|---|---|---|
| **Address is the strongest signal, name is often unreliable.** | S1 "KD Global Flexible Inc" has a true S3 match named "Veoarcnex Labs" at the same address. "Daex LLC" has S2 match "Dimhax LLC" (DBA name). | Name-only matching hits a ceiling. The model must trust address evidence and know when an address is unique. |
| **Strict one-to-one (verified):** 7.64M matched IDs, 0 appear under more than one S1. | Ground-truth scan. | Every S2/S3 record has at most one owner. This turns matching into a global assignment problem, which is a large precision gain. |
| **About 27% of S2 and S3 records are decoys (verified):** S2 3.69M of 5.03M matched, S3 3.94M of 5.29M matched. | Ground-truth scan. | Each record needs a "belongs to nobody" option, not only a best-S1 choice. |
| **Records come in sub-clusters.** For "Currier and Ward", S2 and S3 both carry house number 514 while S1 has 508. For "Aurova", every S3 record has 2658 while S1 has 2657. | Sampled clusters. | Noisy copies mirror each other, so S2/S3 records form tight groups. Cluster them first, then link the group to S1. |
| **City is unreliable, state and street are stable.** | Williamsville vs Buffalo, Boston vs Charlestown, Hampton vs York County. | Do not block on city. Use state, street tokens and house number. |
| **House numbers are noisy** (2657→2658, 247→47, 3280→3280 1/2, HN 974 inserted). | Sampled clusters. | Treat numbers as soft evidence. Do not use exact-match blocking keys. |
| **Name noise types:** legal-suffix moves ("Inc Mcleod Communications"), junk prefixes (`>>`, `...`, `<<`, "THE"), URLs ("marnievelascovaluence.com"), leetspeak ("Transp0rt"), truncation ("Currier And Ward"), typos, embedded numbers, native-script transliteration. | Sampled clusters. | Needs a dedicated name-cleaning stage and a model robust to all of these. |
| **India:** S2/S3 names are often in Devanagari, Gujarati, Telugu or Odia, with addresses in near-verbatim English plus a native-script state. | Sampled clusters. | Address is nearly copy-paste, so the address channel is very strong there. Names need a multilingual model. |
| **France is test-only** (259k S1, 703k S2, 731k S3). | Dataset docs. | Generalization is scored directly. Avoid country-specific features. Adapt with unlabeled test data. |
| Structure (verified): 5.6% of S1 are singletons. The common cluster shape is (1 S2, 1 S3) at 12%, then (1,2) at 11%, (2,1) at 10%. | Ground-truth scan. | Strong priors on cluster size. Feed count-aware features to the final stage. |

---

## 1. Core idea (what makes this different from the standard blocking + XGBoost pipeline)

**Four layers, each adding precision or recall that the standard pipeline lacks:**

1. **Clean, structured, language-agnostic normalization**, with abbreviation dictionaries *learned from the train ground truth* (no external data).
2. **Multi-view hybrid retrieval** (address view, name view, joint view, fuzzy key view) merged with reciprocal-rank fusion, plus retrieval in both directions (S1→S2/S3 and S2/S3→S1).
3. **Cascade of learned scorers:** cheap GBDT pruner, then fine-tuned multilingual **cross-encoder**, then LightGBM stacker with *group-context features* (address rarity, competing candidates, margin over the runner-up).
4. **Collective inference:**
   (a) cluster S2/S3 records first,
   (b) enforce one-owner-per-record with a "nobody" option,
   (c) pick each S1's output set by maximizing **expected F0.5** directly, including the option of an empty list.

Layers 3 and 4 are where most teams stop early. The expected-F0.5 decision rule (4c) matters because the metric is macro per entity and singletons are all-or-nothing.

---

## 2. Preprocessing (details)

### 2.1 Text normalization (all countries, language-agnostic)
1. Unicode NFKC, lowercase, strip zero-width and control characters.
2. Two views of every string. **Raw-fold** keeps the native script. **Latin-fold** is a romanized version (diacritics stripped; Indic scripts romanized with a rule table). Keep both, because the encoder gets the raw form and the fuzzy features get the folded form.
3. Punctuation and spacing: collapse repeated spaces, remove bracket noise (`((INC))`, `>>`, `...`, `<<`, leading `#`).
4. Digit and leetspeak repair: `0→o`, `1→l`/`i` only inside alphabetic tokens (`Transp0rt` → `transport`).
5. URL names: `xyz.com` becomes a token sequence via a dictionary-free word segmenter, using token frequencies counted from the train names themselves.
6. Names: strip `the`, junk symbols, embedded phone numbers, `#12345` store numbers (store as a side feature, not text).
7. Legal-suffix canonicalization: build the suffix vocabulary *from the data* by finding the most frequent last tokens across S1 names and grouping variants that co-occur in matched pairs. Examples: inc / inc. / incorporated / lnc → `INC`; pvt / private / privaet → `PVT`; sarl / sasu / eurl / sci for France appear in test names, so cluster them from the unlabeled test set with the same procedure. Store the suffix class as a categorical feature and remove it from the core name.
8. Word-order invariance: also keep a sorted-token core name ("Inc Mcleod Communications" equals "Mcleod Communications Inc").

### 2.2 Address parsing
Extract into fields: `house_number`, `unit`, `street_tokens`, `locality_tokens`, `state`, `postal`, `landmark_tokens` (Near, Opp, Behind).
- State canonicalization: learn the alias map (NY = New York = `న్యూయార్క్`-style native forms, MH = Maharashtra = महाराष्ट्र) from co-occurrence in train ground-truth pairs. Align tokens that differ but always appear in the same slot, and keep pairs with high count. No external gazetteer.
- Abbreviation map (St = Street, Rd = Road, R. = Rue, Av = Avenue) learned the same way. For France there are no labeled pairs, so mine pairs from *pseudo-labeled* test clusters (section 6.4).
- Numeric handling: keep a list of numbers per address, so fractions ("3280 1/2") and prefixes ("H.no 348", "Door No 348") map to comparable values. Use `abs(number difference)`, `same digits with edits`, and `prefix/suffix containment` as features.
- Token bag with IDF weights: the IDF is computed per country from the union of all sources, so rare tokens (street names) dominate over common ones (road, floor).
- Missing address (about 3.4% of S2 and S3): keep a `missing_addr` flag. These records cannot be matched by address, so a cluster mate's address may be borrowed (section 5.1).

---

## 3. Candidate generation (recall ceiling)

Search **within country** only, with `country` treated as an open string label.

| View | Representation | Index | Catches |
|---|---|---|---|
| A. Address-sparse | word + character-n-gram TF-IDF (BM25-style) on normalized address, IDF-weighted, with the house number as its own token | Sparse matmul in chunks, or a GPU sparse top-k | Copy-paste addresses, typos, reordered components |
| B. Name-dense | fine-tuned multilingual encoder embedding of the cleaned name | FAISS-GPU (IVF-PQ or flat on GPU) | Transliteration, abbreviations, truncation |
| C. Joint-dense | same encoder on `name [SEP] address` | FAISS-GPU | Mixed evidence |
| D. Fuzzy keys | (state, first two street tokens, number-bucket) and (name core sorted tokens) | Hash join | Cases dense retrieval misses. Cheap. |

- Pull top-K per view (start K=30 each), merge with **reciprocal-rank fusion**, cap at about 60 candidates per S1 per source.
- **Reverse retrieval:** for each S2/S3 record, also retrieve its top-10 S1 owners, and add those pairs to the candidate set. This helps recall for decoy-resistant matches and gives the "competition" features in section 5.
- **Target:** at least 97% blocking recall on the held-out train slice. Log recall per view and per country, and compare the union against each view alone. Write the final union to `candidate_pairs.tsv`.
- **Memory note:** 1.73M S1 × 120 candidates = about 200M pairs. Store as int32 pairs in Parquet shards, process per country and per shard, and never build the full pair table in memory.

---

## 4. Models

### 4.1 Encoder (bi-encoder), fine-tuned
- Base: **`BAAI/bge-m3`** (MIT) or **`intfloat/multilingual-e5-base`** (MIT). Both are multilingual and within the size limit.
- Fine-tuning: contrastive InfoNCE using train ground-truth pairs (S1 record vs each matched S2/S3 record).
- **Hard negatives:** (i) different S1 entities at the same street, (ii) same normalized name in a different city, (iii) top-ranked false candidates from the pre-fine-tuning retrieval. This is the step that separates strong from average.
- Train on about 500k to 1M clusters. Mixed precision, in-batch negatives with a large batch, and a lightweight name-only and address-only auxiliary loss so the views specialize.

### 4.2 Stage-1 pruner (fast GBDT)
LightGBM on cheap features for all pairs, keeping the top 8 to 12 candidates per S1:
- Retrieval scores and ranks per view, RRF score.
- rapidfuzz: token_set, token_sort, partial ratio, Jaro-Winkler on name and on address.
- Number features (equal, abs difference, containment), state match, postal match, suffix class match.
- Length ratios, missing flags, script class (Latin, Devanagari, other).

### 4.3 Cross-encoder (the main quality driver)
- Backbone: `xlm-roberta-base` or `mdeberta-v3-base` (both MIT), or the `bge-m3` backbone. Stay under the 8B limit (these are far below it).
- Input format with field tags:
  `[NAME] a-name [ADDR] a-address [SEP] [NAME] b-name [ADDR] b-address [META] country, suffix-class, source`
- Trained on pairs from the stage-1 survivors, including many negatives that look correct on name but not address, and the reverse (same address, different business). The model must learn both directions of the "DBA name, same address" phenomenon.
- Max length 96 to 128 tokens. Estimated inference over about 17M pairs at a few thousand pairs per second on one modern GPU is on the order of an hour. Measure throughput first.

### 4.4 Final stacker
LightGBM over: cross-encoder logit, bi-encoder cosines, stage-1 features, plus **group-context features** (this is the key step):
- Rank and margin of this candidate against the best other S1 that also lists this record.
- Number of S1 entities that share this address (address exclusivity).
- The record's best score to any S1 (a low value suggests it is a decoy).
- Cluster-level features: how many S2/S3 mates are also similar to this S1, and their mean and max scores.
- Country and script class only as *categorical robustness checks*. Drop them if they hurt France-style validation (section 7).
- Calibrate the output with isotonic regression on the held-out split, so that "0.8" means 80%.

### 4.5 Optional, if time allows
A small LLM (Apache-2.0, for example Qwen 1.7B to 4B with LoRA) as a second cross-encoder on the ambiguous probability band only. Ensemble with the encoder-based cross-encoder. Skip unless everything else is finished.

---

## 5. Collective inference (the differentiator)

### 5.1 Mention clustering among S2/S3
Within each country, link S2/S3 records that share a near-identical address plus a compatible name (high stacker probability between them). Form small connected groups. This gives:
- **Borrowed addresses:** an address-less record inherits its group's address for scoring.
- **Group evidence:** a "Veoarcnex Labs" record with a matching address and group-mates named "KD Global" is scored with the group's name evidence.
- Guard against co-located businesses: only merge two records when both name and address evidence agree, and cap the group size.

### 5.2 Ownership assignment
Each S2/S3 record chooses among its candidate S1 owners *plus a "nobody" option*:
- Softmax over the calibrated logits across its competing S1 candidates, with a learned "nobody" logit from the decoy-score feature.
- Keep the assignment only if the top owner wins by a margin. Resolve ties by mutual best match. This uses the verified one-owner rule, so it removes most cross-entity false merges.

### 5.3 Expected-F0.5 output selection (per S1)
The metric per entity is `F0.5 = 1.25·TP / (0.25·T + P)` where T is the true set size and P the predicted set size, and empty prediction against a true singleton scores 1.
- Given calibrated probabilities `p_i` for the candidates, sort descending. For each cutoff k = 0..K, estimate the expected F0.5 by Monte-Carlo (about 200 draws of Bernoulli(p_i)) or a Poisson-binomial DP.
- Choose the k that maximizes it. k=0 is the empty list, whose expected score is P(no true match).
- This replaces a single global threshold. Then compare against the best global threshold on validation, and keep whichever scores higher.

---

## 6. Training and validation protocol

1. **Split by S1 entity** (never by pair), with about 100k S1 held out, sampled by country, and never touched until model selection ends. Keep a second 100k for final calibration.
2. Implement the exact metric (macro per-S1 F0.5, singletons included) and report it overall and per country and per script.
3. **Robustness for France (zero-shot):**
   - Leave-one-state-out check: train without one Indian state or one US region and evaluate on it, as a proxy for an unseen country.
   - **Pseudo-labeling on test France:** take only very high-confidence pairs (exact match of street and number plus near-identical name), mine French abbreviations and suffixes from them, and optionally fine-tune the encoder on them for one epoch. This uses only the provided data.
   - Print sample French outputs and inspect them by eye.
4. **Leakage checks:** no S2/S3 record present in both the training and validation sides.
5. **Ablations** to log: each view of retrieval, cross-encoder on/off, group features on/off, assignment on/off, expected-F0.5 vs global threshold. These go into the methodology document.

---

## 7. Timeline (25th to 27th September; use the AWS $200 and the 48-hour $100 for GPU time)

| When | Work | Exit criterion |
|---|---|---|
| **Day 1 first 5 hours** | Loader, normalizer, address parser, metric. Views A and D plus stage-1 GBDT and a global threshold. **Submit a baseline.** | Valid leaderboard score exists. |
| **Day 1 evening** | Fine-tune the bi-encoder with hard negatives (start overnight). Build views B, C. | Blocking recall at least 95% on validation. |
| **Day 2 morning** | Stage-1 pruner on the union. Build the cross-encoder training set. Start cross-encoder training. | Cross-encoder beats GBDT alone on validation. |
| **Day 2 afternoon** | Group-context features, stacker, calibration, ownership assignment, expected-F0.5. Second submission. | Leaderboard gain is clear. |
| **Day 2 evening** | France pseudo-labeling and abbreviation mining. Error analysis of false merges. | France samples look sane. |
| **Day 3 morning** | Tune on error analysis. Ablations. Final inference over the full test set. | Both TSVs pass the validator. |
| **Day 3 afternoon** | Documentation, README, pinned `requirements.txt`, package the zip. Upload. | Zip matches the required layout. |

Keep a submission-ready fallback at every step. The baseline from the first five hours is the safety net.

---

## 8. Compliance checklist (do not lose points on rules)

- Only MIT or Apache-2.0 models, at most 8B parameters. Record each model's name and license in the documentation.
- No external lookups: no geocoding, no APIs, no external datasets. Pretrained weights are the only outside artifact. The learned dictionaries above come from the provided data only.
- Every S1 test entity gets exactly one row. IDs in results must be S2/S3 IDs that exist. No duplicates within a list.
- `candidate_pairs.tsv` must be exactly what the final scorer saw, and matches must be a subset of it.
- Run: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
- The Unstop page asks for a 1 to 2 page approach document and a zip of code. The problem statement asks for the full template and both TSVs. Keep the document short and follow the problem statement's layout.

---

## 9. Main risks

| Risk | Mitigation |
|---|---|
| Time. Three days for a large pipeline. | Baseline first, every stage independently shippable, ablation-driven decisions. |
| Cross-encoder inference too slow on about 17M pairs. | Tighten the pruner to the top 5 to 8 candidates, shorten max length, batch by length, use fp16. |
| France generalization drops the score. | Language-agnostic features, pseudo-labeling, conservative expected-F0.5 for France, manual inspection. |
| Over-merging co-located businesses. | Group cap, one-owner assignment, precision-first calibration. |
| Calibration drift between train and test. | Calibrate per country. Compare the predicted matched fraction on test against the train prior (about 94% of S1 have at least one match). |

A caveat on the goal: no plan can guarantee beating every other team. This one puts effort into the pieces the metric rewards most (precision, singletons, exploiting the one-owner structure) and the numbers from the ablations will show where it is actually ahead.
