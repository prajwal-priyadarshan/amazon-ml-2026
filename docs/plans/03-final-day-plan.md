# Plan v5: the final day (27 Sep 2026, deadline 23:59 IST)

Status: plan only, nothing here has been run. Written 08:45 IST. Everything below must
finish by **21:00 IST**, which leaves three hours of buffer before the portal closes. The
portal allows 5 submissions per day, and today is the last day.

---

## 0. Honest target

| | score |
|---|---|
| v2, own holdout | 0.9547 |
| **v4 (`output4/`), portal** | **0.97** |
| v4, own holdout | 0.9776 |
| Oracle ceiling on v4's candidates (perfect ranker) | 0.9987 |
| Best public repo for this challenge ([Akash-bardia](https://github.com/Akash-bardia/amazon-ml-challenge-2026)) | 0.9761 validation, portal not stated |
| Next public repo ([mayankgoplani431-del](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution)) | 0.9776 validation → **0.964 portal**, and a 0.9806-validation variant fell to **0.956** |

"Near 100%" is not reachable today. The ceiling on our own candidates is 0.9987, but that
assumes a perfect ranker. Part of the remaining error sits in pairs that no model can tell
apart: a record with no address whose name is shared by other businesses. The realistic goal
for today is **0.980 to 0.985 on the portal**. That would beat every public number above.

**The biggest risk today is overfitting to validation, not a weak model.** One public team
gained +0.003 on validation and *lost* 0.008 on the portal. Every step below is therefore
gated on a validation set built to look like the test set (section 2), not on the plain
holdout.

---

## 1. What we got wrong (measured, not guessed)

| # | Mistake | Evidence | Cost |
|---|---|---|---|
| M1 | **We validate on train-shaped data, but the test set is denser in decoys.** | Train: 10.32M pool records / 2.207M S1 = **4.68 per S1**, 26% of them decoys (records with no owner). Test: 9.97M / 1.733M = **5.75 per S1**. At the train cluster mean of 3.46 matches per S1, about **40% of test records are decoys**. The public repo above reports the same shift (5.8 vs 4.67). | Most of the 0.9776 → 0.97 gap. False positives that the holdout never shows us. |
| M2 | **France is unhandled.** | 15% of test S1, zero labels, never seen by any model, calibrated with the global fallback. The `pseudo` stage exists and was never run. | If India and US score 0.9776 on the portal, France is about 0.93. That is inferred, not measured. |
| M3 | **The same data trained every model.** | Encoder, pruner, judge and stacker all trained on the one fit split. The tuned threshold came out at 0.85, and the per-S1 "expected F0.5" rule lost to a plain threshold. Both are signs of over-confident probabilities. | Poorer calibration, and a decision layer that cannot use per-entity reasoning. |
| M4 | **Records with no address are scored against S1 alone.** | 3.9% of true pairs, yet **63% of all misses**. Only 36% are matched, and about 8,900 wrong no-address pairs also score ≥ 0.5. The v1 plan's "borrow the group's address" (winning_plan §5.1) was never built, and sibling groups are within-source only. | About 1 point on holdout. |
| M5 | **The judge was undertrained.** | One epoch of the cross-encoder, with the loss still falling when training stopped. | Unknown, likely a few tenths. |
| M6 | **Process mistakes.** | An out-of-memory crash on a 47M-pair block cost about 3 h. Time estimates were off by 2 to 3 h. | Time. Today has no slack for this. |

---

## 2. Step 1: build a validation set that looks like the test set (09:00 to 10:00)

Nothing else is trustworthy until this exists.

1. **Measure the shift on test.**
   - For each test S1, count the "look-alike" candidates. A look-alike is a pool record with a high name *or* address similarity where the house number or the name core disagrees with the best candidate.
   - Compute the same count for holdout S1.
   - Compare the two distributions, per country.
2. **Reweight the holdout.**
   - Weight each holdout entity so that its look-alike distribution matches test (histogram ratio, clipped to [0.2, 5]).
   - Report **both** the plain and the weighted holdout F0.5 from now on.
   - The weighted number is the one that gates decisions. It should come out at or below 0.97 for v4. If it does, the proxy reproduces the portal gap and can be trusted. If it doesn't, the shift is somewhere else, and step 3 (the prior shift) becomes the main lever.
3. **France proxy.** Leave-one-country-out: train the stacker on US only and score India. This is the only measurable stand-in for "a country the model never saw", and it gates the France work in section 6.
4. **Split the holdout into two fixed halves** (even and odd S1). Tune on one half and report on the other, every time.

**Gate:** weighted v4 holdout ≈ portal 0.97 (within ±0.005). If it isn't, stop and re-diagnose before spending GPU time.

---

## 3. Step 2: correct probabilities for the test set's lower match rate (10:00 to 10:45, no training)

This is the cheapest lever with the largest expected effect on M1.

The test set has a lower rate of true pairs among its candidates, because it has more decoys. A classifier calibrated on train is **systematically over-confident** on test. This is a textbook label-shift problem, and there is a principled fix ([Saerens, Latinne & Decaestecker 2002, "Adjusting the outputs of a classifier to new a priori probabilities"](https://doi.org/10.1162/089976602753284446)):

- Estimate the test prior π_test from test's own scored probabilities with the Saerens EM loop, per country. Cross-check it against the implied 6.0M matched / 9.97M pool.
- Rescale each calibrated probability: `odds' = odds × (π_test/(1−π_test)) / (π_train/(1−π_train))`.
- Re-run the decision layer on the adjusted probabilities.

**Gate:** does the weighted holdout improve when the same adjustment is simulated? If it does, **submission #1**: v4 plus the prior shift. This is the fastest honest read of whether M1 is real, and it costs no retraining.

---

## 4. Step 3: honest stacker and new features (10:45 to 14:00)

### 4a. Fresh split "fit2" (CPU, about 1 h)

- Take 10% of the train S1 that no model has seen. It must not overlap fit or holdout.
- Run lexical retrieval, dense search (the pool embeddings already exist, so only the S1 side needs embedding, a few minutes), union, prune, and the judge (about 6 min).
- **Only** the stacker and the calibrator train on fit2. This fixes M3.

### 4b. New features (added to the stacker only, never to retrieval)

Ranked by expected value. Every feature is mined from the provided files only.

1. **Cluster consensus (targets decoys).** True copies mirror each other; the v2 data analysis found S2 and S3 both carrying house number 514 while S1 has 508. For each candidate, compare it with the entity's *other* confident candidates in **both** sources:
   - house-number agreement with the consensus;
   - name-core agreement;
   - the fraction of confident siblings it agrees with.

   A decoy disagrees with the consensus; a true copy agrees with it.
2. **Borrowed address (targets M4).** For a record with no address, take the address of its best-matching confident cluster-mate (cross-source, same S1) and compute the address features from that. This is winning_plan §5.1, finally built.
3. **Name rarity.** How many S1 entities and pool records in the same country share this name core (and this name plus city). A name-only match on a unique name is safe; on "Sai Enterprises" it is not.
4. **Record-side "no owner" score.** For each pool record: its best score over *all* S1, the margin to the second-best, and how many S1 claim it. This gives every record an explicit "belongs to nobody" option (winning_plan §5.2), which the test set's 40% decoy rate makes much more important.
5. **House-number conflict, strict version.** Both sides have a number and they differ, as distinct from one side missing. The public repo's 0.9924 precision leaned on exactly this flag.

**Gate:** the new stacker must beat v4 on the **weighted** holdout by at least +0.002, reported on the half it was not tuned on.

---

## 5. Step 4: a stronger judge (GPU, 11:00 to 14:00, in parallel with 4b)

- Continue training the existing cross-encoder (`work3/models/ce`) for **one more epoch**, not from scratch. The training set is:
  - the current **false positives** and high-scoring decoys from fit;
  - **no-address positives**, oversampled 3×;
  - **France pseudo-labels** (section 6), at most 15% of rows.
- Put the borrowed address from 4b into the judge's input for no-address records, marked as borrowed.
- Re-run the judge on fit2, holdout and test: about 45 min, measured from the last run (35 min for fit, holdout and test combined).

**Gate:** judge AUC on the uncertain band must rise on holdout. Otherwise keep the v4 judge and move on. There is no time to debug it.

---

## 6. Step 5: France (runs alongside, 10:00 to 15:00)

1. **Normalization, mined from test France records only.**
   - Accent folding: already done for the lexical views; check that it also applies to features.
   - Abbreviations: `R.`/`Rue`, `Av.`/`Avenue`, `Bd`/`Boulevard`, `Pl.`, `Imp.`, `Chem.`, `Rte`.
   - Legal forms: `SARL`, `SAS`, `SASU`, `EURL`, `SA`, `SCI`, `Ets`.
   - Postal code: a 5-digit code whose first 2 digits are the department.
   - Region ↔ department aliases (`LILLE, Hauts-de-France` = `LILLE, Nord`): mined from pseudo-labelled clusters, as data_analysis_v2 §3 describes, never from a gazetteer.
2. **Pseudo-labels.**
   - Positives: calibrated p > 0.98, *and* mutual best match, *and* agreement on postal code plus house number.
   - Negatives: the same S1's candidates below 0.3.
   - Check that the France statistics of the pseudo-labelled set match the train priors (matches per S1, singleton rate about 5.6%).
3. **France calibration.** Fit France's isotonic curve on the pseudo-labels plus the prior shift from step 2, instead of the global fallback.

**Gate:** the leave-one-country-out proxy (train US, score India) must improve with the same recipe. Then **submission #3**. The France difference shows up only on the portal, and the portal score is the only direct measurement of France.

---

## 7. Step 6: decision layer (15:00 to 16:30)

- Re-run tune with honest probabilities: **expected-F0.5 per S1** against a threshold, per country. With calibrated probabilities the expected-F0.5 rule should win now; it lost before because of M3.
- Keep one-owner, and add the record-side "nobody" option from 4b.4.
- Consider separate thresholds per source. The public 0.976 repo uses τ_S2 = 0.75 and τ_S3 = 0.85. We tried separate thresholds for no-address records and got no gain, so this is a low priority.
- Tune on the even half and report on the odd half, both weighted.

---

## 8. Submission plan (5 left today)

| # | When | What | What it tells us |
|---|---|---|---|
| 1 | about 11:00 | v4 + prior shift (step 2) | whether decoy over-confidence (M1) is real |
| 2 | about 13:00 | optional probe: v4 with France left empty | France's true portal score, from the difference |
| 3 | about 16:30 | honest stacker + new features + France work | the main candidate |
| 4 | about 19:00 | #3 + retrained judge | whether the judge pays off |
| 5 | by 22:00 | the final pick | — |

**Final pick rule.** Choose the best **weighted-holdout** candidate. Where two candidates are within 0.002 of each other, prefer the one with the better portal score. Never pick a portal gain that the weighted holdout contradicts: the private leaderboard is a different subset, and the public repo's 0.9806 → 0.956 shows what chasing a single number does.

The final pick must also pass `validate_submission.py` with no "matched ID outside candidates" warning, and it must go into the submission zip with its `candidate_pairs.tsv` and code.

---

## 9. Timeline (IST)

| Time | GPU | CPU | Me |
|---|---|---|---|
| 09:00–10:00 | idle | look-alike stats, weighted holdout | step 1 |
| 10:00–10:45 | idle | prior-shift EM | step 2 → **submit #1** |
| 10:45–11:45 | fit2 S1 embed, then build CE training set | fit2 retrieval + prune | 4a, France normalization |
| 11:45–14:00 | **train-ce, +1 epoch** | new features on fit2, holdout and test | 4b, France pseudo-labels |
| 14:00–14:45 | `ce` on fit2, holdout and test | — | — |
| 14:45–16:30 | idle | stacker, calibration, tune | 6, 7 → **submit #3** |
| 16:30–19:00 | buffer or a second judge seed | errors, ablations | **submit #4** |
| 19:00–21:00 | — | final resolve, validate, zip, documentation | **submit #5** by 22:00 |

**Memory rules learned the hard way:**
- one (country, source) block at a time;
- score in blocks of 4M rows or fewer;
- run every job longer than 10 minutes in a detached window with its log written to a file.

**Cut order if behind:** drop the second judge epoch (section 5) first, then France pseudo-labels (keep the normalization and the prior shift), then the borrowed-address feature. **Never cut** the weighted holdout (step 1) or the prior shift (step 2).

---

## 10. What we will not do today

- **No new retrieval.** Recall is 0.9958 and only 0.6% of true pairs are lost before ranking.
- **No bigger encoder** (e5-base, xlm-r-large). A 3× slower judge does not fit in the time left.
- **No external data of any kind:** no gazetteers, no geocoding, no public business lists. That is grounds for disqualification.
- **No tuning on the portal.** It is feedback, not a validation set.

---

## Sources

- Public repos for this challenge: [Akash-bardia (0.9761 val)](https://github.com/Akash-bardia/amazon-ml-challenge-2026), [mayankgoplani431-del (0.9776 val → 0.964 LB; test distractor density 5.8 vs 4.67)](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution)
- Prior shift: Saerens, Latinne & Decaestecker, *Neural Computation* 14(1), 2002, "Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure"
- Cross-encoder matching: [Ditto, Li et al., VLDB 2021](https://arxiv.org/abs/2004.00584), which reports 96.5% F1 on 789K × 412K company records
- Collective / cluster-level ER and the risk of transitive closure: [Bhattacharya & Getoor, TKDD 2007](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf), [Entity Resolution in Practice (2026)](https://arxiv.org/pdf/2607.26298)
- Our own measurements: `work3_neural.log` (diag, score, errors), `docs/data_analysis_v2.md`, `docs/winning_plan.md` §5
