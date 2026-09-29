<div align="center">

# Business Entity Resolution

### Amazon ML Challenge 2026 &nbsp;·&nbsp; Team Wizards

*Matching 1.7M business records against ~10M noisy candidates — retrieve, rank, resolve.*

<img src="https://img.shields.io/badge/Python-3776AB-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"> <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"> <img src="https://img.shields.io/badge/LightGBM-02569B?style=flat-square" alt="LightGBM"> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn"> <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas"> <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"> <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy"> <img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA"> <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle">

<br>

[**Overview**](#overview) &nbsp;|&nbsp; [**Results**](#results) &nbsp;|&nbsp; [**Approach**](#approach) &nbsp;|&nbsp; [**Run it**](#run-it) &nbsp;|&nbsp; [**Explore the repo**](#explore-the-repo) &nbsp;|&nbsp; [**Team**](#team)

<br>

<table>
  <tr>
    <td align="center"><h2>0.985018</h2><sub>portal score</sub></td>
    <td align="center"><h2>#525</h2><sub>final rank</sub></td>
    <td align="center"><h2>0.9862</h2><sub>held-out macro F0.5</sub></td>
    <td align="center"><h2>24.2M</h2><sub>rows normalized</sub></td>
    <td align="center"><h2>+550</h2><sub>ranks climbed</sub></td>
  </tr>
</table>

</div>

---

## Overview

Given a deduplicated reference table **S1** (a business name, an address, a country), find
every record in two large, noisy pools **S2** and **S3** that describes the same business.
About 27% of S2/S3 records belong to no S1 entity at all — pure decoys — and every real
match is a strict one-to-one: no pool record is ever claimed by two S1 entities. Scored by
**macro-averaged F0.5 per S1 entity**, singletons included.

The test split alone is 1,732,544 S1 entities matched against 9,969,589 pool records — no
subsampling anywhere, because a subsampled negative pool reads a few points high and
doesn't survive contact with the real leaderboard.

We didn't win, but the pipeline got within striking distance of the best public numbers:
three architecture generations in five days, a 24-million-row dataset, and a cluster-aware
refine model built the night before the deadline. This repo is that pipeline, reorganized
after the competition closed.

## Results

### Score, generation by generation

| generation | what changed | held-out macro F0.5 |
|---|---|---|
| v1 | word/char TF-IDF retrieval, two-stage GBDT | ~0.90 (portal) |
| v2 | multi-view retrieval, isotonic calibration, Monte-Carlo expected-F0.5 selection | **0.9547** — kept at [`results/baseline_v2/`](results/baseline_v2/) |
| v3 (tabular) | five sparse retrieval views + LightGBM pruner/stacker | 0.9480 |
| v3 (neural) | + fine-tuned multilingual e5-small bi-encoder retrieval, XLM-R cross-encoder judge | **0.9776** — kept at [`results/neural_v3/`](results/neural_v3/) |
| v3 + refine | + cluster-aware refine model trained on held-out splits | 0.9822 |
| v3 + refine, full matrix | + full-S1-table name rarity, legal-form and IDF features | 0.9843 |
| **final** | + two extra clean splits, a second judge epoch, 3-seed bagging | **0.9862 / 0.9886** (open/closed) — **portal 0.985018**, kept at [`output/`](output/) |

The single biggest lever was retrieval: fine-tuning the bi-encoder on the competition's own
matches (InfoNCE with mined hard negatives) lifted pair recall from 93.2% to 99.6% and moved
the oracle ceiling from 0.974 to 0.999. Everything downstream of that — the judge, the
stacker, the refine model — was fighting over the remaining 1.3 points.

### Leaderboard climb

<table>
<tr>
<td width="55%" valign="top">

| Checkpoint | Rank |
|---|---|
| Start, afternoon of 26 Sep | ~#1,080 |
| Low, after midnight 27 Sep | ~#1,900 |
| Peak, evening of 27 Sep | ~#290 |
| **Final, 12:04 AM 28 Sep** | **#525** |

18 checkpoints, **+550 ranks** net. The rank drifted back from the evening peak even though
the score held steady, which is what a still-moving leaderboard looks like when other teams
land late improvements.

</td>
<td width="45%">

<img src="docs/assets/leaderboard-rank-timeline.png" alt="Rank timeline across 18 leaderboard checkpoints, ending at rank #525 with score 0.985018">

<sub>Source: Amazon ML Challenge Explorer</sub>

</td>
</tr>
</table>

## Approach

```
 S1 × (S2 ∪ S3)        RETRIEVE              PRUNE               JUDGE               RESOLVE
  ~24M records     ┌───────────────┐   ┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
 ───────────────►  │ 5 sparse views│──►│ LightGBM keeps│──►│ XLM-R cross-    │──►│ per-S1 expected │
                   │ + fine-tuned  │   │ top 8 per     │   │ encoder judge   │   │ F0.5 pick, one  │
                   │ e5-small      │   │ S1/source,    │   │ (uncertain band)│   │ owner per       │
                   │ bi-encoder    │   │ sibling       │   │ + stacker       │   │ record, then a  │
                   │               │   │ expansion     │   │ + refine        │   │ cluster-aware   │
                   │               │   │               │   │                 │   │ refine pass     │
                   └───────────────┘   └───────────────┘   └─────────────────┘   └─────────────────┘
                     ~202M pairs         9.2M pairs          calibrated             final matches
                                         survive             probabilities
```

The last piece is the one that mattered most: a **cluster-aware refine model trained only
on entities no upstream model ever saw**. Every earlier stage — the fine-tuned encoder, the
cross-encoder judge, the stacker — is trained on the same split it's evaluated on, which
overstates confidence. The refine model instead trains on three *held-out* splits (850k
fresh entities, 3.9M pairs, S1-grouped cross-fitting) and scores every candidate against the
consensus of the entity's *other* confident copies — same name, same address, same house
number, same legal form — rather than in isolation.

Full write-up: [`docs/methodology.md`](docs/methodology.md) (submitted for judging) and
[`docs/architecture.md`](docs/architecture.md) (the design rationale).

## Run it

The dataset (`student_resource/`) is provided by the organizers and isn't tracked. Drop it
in at the repo root, then:

```bash
pip install -r requirements.txt          # tabular stages only
# neural stages additionally need torch + requirements-v3.txt — see docs/runbook.md
```

Follow [`docs/runbook.md`](docs/runbook.md) for the current (v3) pipeline, stage by stage.
[`docs/legacy-v1v2-pipeline.md`](docs/legacy-v1v2-pipeline.md) covers the lighter v1/v2
fallback, which needs no GPU. Hardware used: 16 GB RAM, RTX 4060 laptop (8 GB VRAM),
Python 3.14. Models: `intfloat/multilingual-e5-small` and `FacebookAI/xlm-roberta-base`
(both MIT), fine-tuned only on the provided training data — no external data.

<details>
<summary><b>Build the submission zip</b></summary>

<br>

One command assembles `submission\<team>_submission.zip` in the organizers' required layout
and runs the validator on it (details in [`SUBMISSION.md`](SUBMISSION.md)):

```powershell
.\tools\build_submission.ps1 -TeamName "YourTeam" -Members "Name One, Name Two"
```

</details>

## Explore the repo

<details>
<summary><b>Repo layout</b></summary>

<br>

Repo root matches the exact `<team>_submission.zip` layout the organizers require (see
[`SUBMISSION.md`](SUBMISSION.md)), plus supporting docs and history:

```
├── README.md                 — you are here
├── SUBMISSION.md             — how the official submission zip is built (now: just zip this repo)
├── Documentation_template.md — the filled-in methodology write-up, exact required filename
├── output/                   — the official deliverable: matching_results.tsv + candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/                  — the pipeline (same code as below, required path for judging)
│   ├── README.md             — reproduction instructions (copy of docs/runbook.md)
│   └── requirements.txt      — pinned dependencies
├── tools/                    — run scripts, build_submission.ps1, the official validator
├── notebooks/                — self-contained Kaggle notebook (no repo dependency)
├── docs/
│   ├── methodology.md        — the write-up submitted for judging
│   ├── architecture.md       — the design plan v3 was built from, with its error budget
│   ├── data-analysis.md      — measured facts about the data that drove every design choice
│   ├── runbook.md            — stage-by-stage instructions for running v3 end to end
│   ├── legacy-v1v2-pipeline.md — how to run the v1/v2 fallback
│   ├── problem_statement.pdf — the organizers' original problem statement
│   └── plans/                — earlier, superseded planning documents, kept for history
├── results/                  — two earlier milestone submissions (baseline_v2, neural_v3)
├── archive/final-day-run-log/ — the literal commands run in the last hours before the deadline
└── student_resource/         — organizer-provided dataset + validator (not tracked; see below)
```

</details>

<details>
<summary><b>What would have closed the gap</b></summary>

<br>

Rank #525 with a 0.985 holdout-consistent score means the top of the leaderboard found
something structural this pipeline didn't. The honest gaps, in order of size:

- **France got zero-shot treatment.** It's 15% of the test set, entirely unlabeled, and
  region↔department address aliasing (`LILLE, Hauts-de-France` = `LILLE, Nord`) was only
  ever mined from pseudo-labelled clusters, never finished with a proper gazetteer pass.
- **No-address records were the single largest miss bucket** — 3.9% of true pairs but 63%
  of all misses before the refine layer's "borrow the group's address" feature, and even
  after it, only about half of no-address records are matched correctly.
- **No global assignment solver.** Matching is closer to optimal per-record greedy
  (each S2/S3 record keeps its single best S1) than a true one-to-many Hungarian-style
  assignment, which was judged unnecessary given a measured max cluster size of 11 — worth
  re-checking against whatever the winning teams did differently.

See [`docs/plans/03-final-day-plan.md`](docs/plans/03-final-day-plan.md) for the plan this
was measured against on the last day, and [`docs/methodology.md`](docs/methodology.md) §5
for the full error taxonomy.

</details>

---

## Team

<div align="center">

<table>
  <tr>
    <td align="center" width="25%"><a href="https://github.com/prajwal-priyadarshan"><img src="https://github.com/prajwal-priyadarshan.png?size=160" width="110" style="border-radius:50%" alt="Prajwal Priyadarshan"></a><br><b>Prajwal Priyadarshan</b><br><a href="https://github.com/prajwal-priyadarshan">@prajwal-priyadarshan</a></td>
    <td align="center" width="25%"><a href="https://github.com/kesavvvvvv"><img src="https://github.com/kesavvvvvv.png?size=160" width="110" style="border-radius:50%" alt="Kesav Satya Sai Nimmagadda"></a><br><b>Kesav Satya Sai Nimmagadda</b><br><a href="https://github.com/kesavvvvvv">@kesavvvvvv</a></td>
    <td align="center" width="25%"><a href="https://github.com/KishoreB25"><img src="https://github.com/KishoreB25.png?size=160" width="110" style="border-radius:50%" alt="Kishore B"></a><br><b>Kishore B</b><br><a href="https://github.com/KishoreB25">@KishoreB25</a></td>
    <td align="center" width="25%"><a href="https://github.com/KKabilan07"><img src="https://github.com/KKabilan07.png?size=160" width="110" style="border-radius:50%" alt="Kabilan K"></a><br><b>Kabilan K</b><br><a href="https://github.com/KKabilan07">@KKabilan07</a></td>
  </tr>
</table>

Built at **Amrita Vishwa Vidyapeetham** for the Amazon ML Challenge 2026.

</div>
