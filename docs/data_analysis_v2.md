> **Reference, still current.** Measured facts about the data, used by
> [plan_v3_architecture.md](plan_v3_architecture.md). Build from that plan, not this file.

# Entity Resolution: measured data analysis and v2 plan

Every number below was measured on the actual files in this session. Nothing is estimated
unless it says so. Measurement scripts are throwaway (scratchpad); the facts are here.

---

## 1. What the task actually is

Given a **deduplicated reference table S1**, find for each S1 row the set of S2 and S3 rows
describing the same business. Three fields only: `business_name`, `business_address`,
`country`. Scored by **macro-averaged F0.5 per S1 entity**, singletons included.

Two measured structural facts change the whole formulation:

| Fact | Measurement | Consequence |
|---|---|---|
| **Strict one-to-one ownership** | 7,638,365 distinct matched IDs in train GT; **0** appear under more than one S1 | This is not independent pair classification. It is a **global assignment problem**: every S2/S3 record has at most one owner. Enforcing this removes a whole class of false merges for free. |
| **(name, address) is a perfect key in S1** | Over 2,206,821 S1 rows: name alone collides with another S1 **39.53%** of the time, address alone **5.46%**, name+address **0.00%** | There is zero irreducible ambiguity on the S1 side. Every hard case comes from noise on the S2/S3 side, never from two genuinely identical S1 entities. A confident (name, address) agreement is therefore *decisive*, and a matcher that uses only one field is throwing away the only exact key that exists. |

Supporting structure:

- **Decoys:** S2 has 5,034,617 rows of which 3,693,619 are matched → **26.6% belong to nobody**.
  S3: 3,944,746 of 5,285,604 → **25.4% decoys**. Every record needs an explicit "no owner" option.
- **Singletons:** 5.58% of S1 have no match at all. Under F0.5 these are all-or-nothing:
  correct empty list = 1.0, any false match = 0.0.
- **Cluster shapes:** 1-1 (12.2%), 1-2 (11.4%), 2-1 (10.1%), 2-2 (9.4%), 0-0 (5.6%), 1-3 (6.4%).
  Mean 3.46 matches, max 11. Cluster size is strongly predictable → feed count-aware features
  to the decision layer.

---

## 2. Which field carries the identity signal

Measured on ~139k true pairs sampled from train ground truth. Jaccard is over
accent-folded alphanumeric token sets.

| | US | India |
|---|---|---|
| pairs scored | 84,102 | 54,730 |
| name matches exactly after normalization | 30.2% | 18.6% |
| name Jaccard ≥ 0.8 | 42.3% | 33.1% |
| **name Jaccard < 0.34 (name is useless)** | **12.8%** | **13.3%** |
| address Jaccard ≥ 0.8 | 16.3% | **47.9%** |
| address Jaccard < 0.34 | 14.4% | 11.3% |
| address empty | 4.7% | 3.8% |
| **digit/house-number overlap** | **87.7%** | **94.4%** |
| ADDR-ONLY (name dead, address rescues) | 5.9% | 8.4% |
| NAME-ONLY (address dead, name rescues) | 10.1% | 6.1% |
| **both channels weak** | **1.7%** | **1.2%** |

### Reading this

1. **Neither field is sufficient alone.** Name is useless on ~13% of true pairs in both
   countries. Address is useless on 11–14%. A single-channel matcher has a hard ceiling.
2. **The house number is the single most reliable token** — 88–94% overlap, far above any
   other signal. It should be extracted as its own field, never left inside a text blob,
   and compared with digit-edit-distance (noise like 2657→2658, 247→47, `3280 1/2` is real).
3. **The two countries have opposite profiles.** India addresses are long (63–78 chars) and
   near copy-paste, so address Jaccard ≥ 0.8 fires half the time. US addresses are short
   (32–39 chars), heavily reordered and abbreviated, so Jaccard collapses to 16% even though
   the address is usually *semantically* identical. The US failure is a normalization failure,
   not a signal failure.
4. **~1.2–1.7% of true pairs have no usable pair-level evidence at all.** These are only
   reachable through group inference (a cluster-mate shares the address). This is the
   empirical ceiling on any purely pairwise system: about **98.5%**.

### Why the name channel dies

Real examples pulled from true matches:

```
S1 'Continental Beacon Home Improvements Inc'  ~  S3 'Deltavantage'
   24331 Old 24, Woodburn, IN                     Woodburn, 24331 Old 24, IN

S1 'Global Constructions Pvt Ltd'              ~  S2 'Rizazepharia'
   Flora 17, Tower B Blk No. 202, S P Road...      (identical string)

S1 'Akshaya Gases Limited'                     ~  S3 'akshayagases.com'
S1 'White Management Pvt Ltd'                  ~  S2 'WM'          (initialism)
S1 'Huff, Mendez and White'                    ~  S3 'Huff'        (truncation)
S1 'Tiify LLC'                                 ~  S2 'TIFY LLC'    (typo)
S1 'Vadyne Inc'                                ~  S2 'Vadyne Incorporated #34301'
```

The name is sometimes a **completely unrelated random string**. No string metric and no
embedding will ever match `Deltavantage` to `Continental Beacon Home Improvements`. Those
pairs are recoverable *only* from the address. Conversely `WM` ← `White Management` and
`akshayagases.com` ← `Akshaya Gases` require initialism and URL-segmentation handling.

### Addresses are permuted component sets, not strings

`Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac` vs
`5 bis Rue Pierre Dignac, La Teste-de-Buch, Nouvelle-Aquitaine`.

Comma-separated components are **shuffled**. Any sequence-sensitive similarity
(Levenshtein, partial ratio, sequence embeddings on the raw string) is actively
misleading. Parse into fields, compare as sets.

---

## 3. Scripts, and what France actually is

| file | country | non-Latin names | non-Latin addresses | empty addr |
|---|---|---|---|---|
| train S1 | US | 0.0% | 0.0% | 0.0% |
| train S1 | India | 0.0% | 0.06% | 0.0% |
| train S2 | India | **27.9%** | 23.7% | 2.9% |
| train S3 | India | **18.5%** | 22.5% | 3.1% |
| train S2/S3 | US | 6.3–6.8% | 0.0% | 2.8–3.7% |
| test S1 | **France** | **15.7%** | **28.3%** | 0.0% |
| test S2/S3 | France | 23.9–24.5% | 24.1–24.3% | 2.9–3.1% |

**S1 is always clean and romanized** for US and India — the noise is entirely on S2/S3.
**France breaks that pattern**: the S1 side itself carries accents. So France is not
"India with a different script"; it is a regime where *both* sides are dirty.

But France is **Latin script with diacritic corruption**, which is far easier than Devanagari:

```
S1  '<< Team Ecole'                      175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine
S1  'ZNB Club SARL'                      Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac
S2  'SCI Ptit Àmicale'                   18 RUE JEN ZAY, Dunkerque, Nord
S2  'OZT ÀMICALE SAS'                    24 R DESAIX, TOURCOING, Hauts-de-France
S2  'Marina Ecole France Sarl'           63 R. DE DIEPPE, LILLE, Hauts-de-France
S3  'Europ & Frères Distribution S.A.'   (41) Rue Des Thuyas, Lège-cap-ferret, Gironde
```

Four France-specific mechanics, all cheap to fix:

1. **Injected diacritics** — `Àmicale`, `Àrt`, `ÀMICALE`. Accent folding is mandatory.
2. **French street abbreviations** — `R.`/`R` = Rue, `AV` = Avenue, `BD` = Boulevard,
   `ALL`/`ALLÉE`, `IMP`, `RTE`, `CHEM`.
3. **French legal suffixes** — SARL, SAS, SASU, EURL, SCI, SNC, S.A., S.A.S, and bracketed
   forms `[EURL]`.
4. **Region vs department mismatch** — `LILLE, Hauts-de-France` and `LILLE, Nord` are the
   same place at two administrative levels (Nord ⊂ Hauts-de-France). Same for
   `La Teste-de-Buch, Gironde` vs `..., Nouvelle-Aquitaine`. This is exactly the
   Williamsville/Buffalo problem in US data. **The last address component must be treated as
   a soft, learnable alias — never as a hard blocking key or an equality feature.**

There is no labelled France data, so the region↔department alias table must be mined from
**pseudo-labelled test clusters**: group France records by (house number + street tokens),
then read off which region and department tokens co-occur.

---

## 4. Blocking: no hard key is good enough

Recall ceiling of each candidate key, measured on the same true-pair sample. This is an
upper bound on what any matcher built on that key can score.

| key | US | India |
|---|---|---|
| exact normalized core name | 55.1% | 44.8% |
| name first-4-chars | 90.5% | 67.5% |
| sorted address digits | 61.4% | 57.2% |
| first address digit | 67.7% | 69.1% |
| **any shared address word** | **95.0%** | **96.1%** |
| shared address word pair | 77.7% | 88.1% |
| first digit + address word | 67.4% | 69.0% |
| **name character 3-gram** | **98.0%** | 80.4% |
| **union of 4 hard keys** | **86.1%** | **83.1%** |

### Reading this

- **Hard-key blocking tops out at ~85%.** Any pipeline whose candidate generation is a set
  of hash joins is structurally capped around 0.85 before the model even runs. If v1 does
  this, that alone explains the score.
- The only keys above 95% are **soft, high-volume** ones: shared address word (95–96%) and
  name 3-gram (98% US, 80% India). These are not usable as hash buckets — they are
  **IDF-weighted retrieval indexes**. Candidate generation has to be top-K retrieval, not
  bucketing.
- **The two high-recall views are complementary across countries**: name 3-gram is the
  strong view for US (98%) and weak for India (80%, Devanagari); address words are strong
  everywhere (95–96%). Running both and fusing is what gets you past 99%.
- Target: ≥99.3% union recall at ≤60 candidates per S1. Measure it per country and per view,
  and log the reduction ratio.

---

## 5. Why v1 scored 0.90 — three testable hypotheses

France is 259,452 of 1,732,544 test S1 rows = **14.97%**. If US+India were performing at
0.97, then `0.8503 × 0.97 + 0.1497 × x = 0.90` gives **x ≈ 0.50**. So a France score around
0.5 fully explains the gap on its own. Competing explanations:

| # | Hypothesis | How to test | Expected fix size |
|---|---|---|---|
| **H1** | France underperforms badly (no accent folding, no French abbreviations, region/department treated as equality) | Leave-one-country-out: train on US, validate on India. If the drop is large, generalization is the problem. | up to +7 pts |
| **H2** | Blocking recall is capped (hard keys ≈ 85%) | Measure candidate recall on a ≥200k-S1 validation slice, per country, per view | up to +10 pts |
| **H3** | Decision rule mis-tuned — fixed threshold 0.7 from `work_dry/config.json`, tuned on a 5,000-entity dry run, plus false merges on the 5.6% singletons | Score validation split by cluster size and singleton/non-singleton | +2–4 pts |

These are cheap to run and they are mutually distinguishable. **Run all three before writing
any new model code** — fixing the wrong one burns days.

The 0.9918 held-out F0.5 in `work_dry/config.json` is not evidence against any of this: it
came from a 5,000-entity dry run, where the S2/S3 pools are ~3,000× smaller than in the real
test set and essentially contain no lookalikes.

---

## 6. Recommended architecture

```
RAW  ──►  NORMALIZE  ──►  RETRIEVE (4 views)  ──►  PRUNE  ──►  CROSS-ENCODE  ──►  STACK
                                                                                    │
                      FINAL OUTPUT  ◄──  EXPECTED-F0.5  ◄──  ASSIGNMENT  ◄──  GROUP
```

### 6.1 Normalization — the highest return-per-hour stage

Not a preprocessing detail: measured above, US address Jaccard is 16% purely because of
reordering and abbreviation. Fixing normalization *is* the model improvement.

- NFKC, casefold, strip zero-width/control chars, fold diacritics (keeps a raw view too).
- Parse the address into `house_number`, `unit`, `street_tokens`, `locality`, `admin_area`,
  `postal`, `landmark` — treating comma components as an unordered set.
- Numbers: keep as a multiset with `3280 1/2`, `H.no 348`, `(41)` all reduced to comparable
  values. Compare with digit edit distance, not equality.
- Street-type canonicalization across all three languages (St/Rd/Ave/Blvd, Rue/Av/Bd/Allée).
- Legal-suffix canonicalization, kept as a **separate categorical feature** and stripped from
  the core name — different suffixes can mean different companies.
- Name repair: leetspeak (`Transp0rt`), junk prefixes (`>>`, `<<`, `...`, `((INC))`), store
  numbers (`#34301` → side feature), URL segmentation (`akshayagases.com` → akshaya gases),
  initialism expansion (`WM` ← White Management), sorted-token view for word-order invariance.
- Alias tables (state, street type, region↔department) **mined from train GT co-occurrence**,
  and for France from pseudo-labelled test clusters. No external gazetteers.

### 6.2 Retrieval — four views, reciprocal-rank fusion, both directions

| view | representation | catches |
|---|---|---|
| A. address sparse | IDF-weighted word + char n-gram TF-IDF, house number as its own token | copy-paste and reordered addresses (95–96% ceiling) |
| B. name dense | fine-tuned multilingual bi-encoder (`bge-m3` / `multilingual-e5-base`) | transliteration, abbreviation, truncation |
| C. joint dense | same encoder on `name [SEP] address` | mixed evidence |
| D. name char 3-gram sparse | TF-IDF | typos; 98% ceiling on US |

Top-30 per view, RRF fusion, cap ~60 per S1 per source. **Also retrieve in reverse**
(each S2/S3 record → top-10 S1 owners): this is what supplies the competition features in
6.4 and the "no owner" evidence for the 26% decoys.

Memory: 1.73M S1 × 120 candidates ≈ 200M pairs. Shard by country, store int32 pairs in
Parquet, never materialize the full pair table.

### 6.3 Scoring cascade

1. **GBDT pruner** on cheap features (retrieval ranks/scores, rapidfuzz token_set/token_sort/
   Jaro-Winkler, digit overlap, suffix match, missing flags, script class) → keep top 8–12.
2. **Cross-encoder** (`xlm-roberta-base` / `mdeberta-v3-base`), field-tagged input:
   `[NAME] … [ADDR] … [SEP] [NAME] … [ADDR] … [META] country, suffix, source`.
   This is the main quality driver. Trained on stage-1 survivors with **hard negatives**:
   (i) different businesses at the same street, (ii) same name in a different city,
   (iii) top-ranked false candidates from pre-finetuning retrieval. Both directions of the
   "same address, different business" / "different name, same business" phenomenon must be
   in the training set, because the data contains both.
3. **LightGBM stacker** over cross-encoder logit + bi-encoder cosines + stage-1 features +
   **group-context features**: margin over the runner-up S1 for this record, address
   exclusivity (how many S1 share this address), the record's best score to *any* S1 (low ⇒
   decoy), cluster-mate agreement. Calibrate with isotonic regression.

### 6.4 Decision layer — where the remaining points are

1. **Group S2/S3 records first.** Noisy copies mirror each other, so records sharing a
   near-identical address plus compatible name form tight groups. This lets an
   address-less record (3–5% of S2/S3) borrow its group's address, and lets the
   1.2–1.7% "both channels weak" pairs be recovered through a mate.
2. **Global assignment with a "nobody" option.** Each S2/S3 record picks at most one S1,
   softmax over its candidate owners plus a learned no-owner logit. Keep only if the top
   owner wins by a margin; resolve ties by mutual best match. This is licensed by the
   measured 0.0000% ownership collision and is the single biggest precision lever.
3. **Per-S1 expected-F0.5 selection** instead of a global threshold. Sort calibrated
   probabilities, and for each cutoff k = 0..K estimate E[F0.5] by Poisson-binomial DP or
   Monte-Carlo. Pick the best k. k = 0 (empty list) has expected score P(no true match),
   which is how the 5.6% singletons get scored correctly. Keep whichever of this and the
   best global threshold wins on validation — `work_dry/config.json` currently shows
   `"mode": "threshold"`, so this path is **not currently active**.

---

## 7. Validation protocol (fix this before anything else)

The 5,000-entity dry run is the reason the leaderboard was a surprise. Replace it with:

1. **Entity-aware split.** Split by S1 entity, and hold out the *whole* cluster. Never let a
   record from a held-out cluster sit in the training pool — that leaks the answer through
   the one-owner constraint.
2. **Full-size pools.** Validation S1 entities must be scored against the **complete** S2/S3
   candidate pool, not a sampled one. Otherwise lookalike density is wrong by orders of
   magnitude and every precision estimate is meaningless.
3. **Leave-one-country-out.** Train on US, validate on India, and vice versa. This is the
   only available proxy for France, and it directly measures H1.
4. **Report the breakdown every time**: F0.5 overall, per country, per cluster size, and for
   singletons alone. Plus candidate recall per view. A single aggregate number hides exactly
   the failure that is costing 9 points.

---

## 8. Priority order

| # | Action | Why | Est. gain |
|---|---|---|---|
| 1 | Build the real validation harness (§7) | Cannot diagnose anything without it | 0, unblocks all |
| 2 | Measure candidate recall per country per view | Tests H2, sets the ceiling | 0, diagnostic |
| 3 | Normalization overhaul: address component parsing, accent folding, French abbreviations, region↔department aliases | US Jaccard 16% is a normalization artifact; France is 15% of the score | **+4–8** |
| 4 | Retrieval: add the two ≥95% soft views, RRF, reverse retrieval | Lifts the 85% hard-key ceiling toward 99% | **+3–6** |
| 5 | Global assignment + expected-F0.5 selection | Licensed by measured one-owner; fixes singletons | **+2–4** |
| 6 | Group/cluster inference before assignment | Recovers the 1.2–1.7% no-evidence pairs and 3–5% empty addresses | +1–2 |
| 7 | Fine-tuned cross-encoder with hard negatives | Main quality driver on the genuinely ambiguous band | +2–4 |

Items 3–5 are the ones that pay for themselves fastest, and none of them require a GPU.
Item 7 does, and should run on Kaggle.

---

## 9. What is not yet known

- The true per-country score of v1. Not observable from the portal; §7.3 is the proxy.
- Actual candidate recall of the current pipeline at full scale. Never measured.
- Whether the France region↔department alias mining works — depends on how clean the
  pseudo-labelled clusters are, which has not been tested.
- Cross-encoder inference throughput over ~17M surviving pairs. Must be measured before
  committing to it in the final run.
