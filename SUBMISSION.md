# Building the official submission zip

The competition requires a `<team_name>_submission.zip` with this exact structure (from the
organizers' submission page):

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv     # final matches, same file uploaded to the leaderboard
│   └── candidate_pairs.tsv      # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                 # all source code
│       ├── README.md            # end-to-end reproduction instructions
│       └── requirements.txt     # pinned dependencies / environment
└── Documentation_template.md    # completed methodology write-up
```

**As of the Sep 2026 restructure, the repo root itself already matches this layout** —
`output/`, `code/business_entity_resolution/`, and `Documentation_template.md` all live at
the top level. There is no staging step anymore: clone (or pull) the repo and zip the four
items above directly.

## Fastest path: `tools/build_submission.ps1`

```powershell
.\tools\build_submission.ps1
```

Zips `output/`, `code/business_entity_resolution/`, and `Documentation_template.md` straight
from the repo root into `submission\<safe-repo-name>_submission.zip`, skipping every other
top-level folder (`work3_*`, the historical `output8/output9/output11.../` experiment
variants, `docs/`, `student_resource/`, etc. — none of those are part of the required zip).
Then runs the validator on the result. Pass `-SkipValidate` to skip that last step.

## Manual path (e.g. for a teammate without the `.venv`)

Just zip the four required top-level entries — any zip tool works, including Windows'
built-in "Compress to ZIP file", as long as it preserves folder structure and doesn't add an
extra wrapping folder:

- `output/`
- `code/`
- `Documentation_template.md`

Rename the result to `<your_team_name>_submission.zip` before uploading.

**Note for teammates cloning fresh:** `output/candidate_pairs.tsv` (135 MB) is tracked via
**Git LFS** — install `git-lfs` and run `git lfs pull` after cloning, or the file in your
working copy will just be a small pointer stub instead of the real data. Everything else is
plain git.

## Validate before uploading

```bash
python3 tools/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  student_resource/dataset/test
```

`PASS` means it's safe to upload. `output/matching_results.tsv` is the file that was already
scored on the leaderboard (0.985018, rank #568) — this confirms your working copy matches
byte-for-byte and `candidate_pairs.tsv` is well-formed.

## Background

Before this restructure, the repo used a public-GitHub-friendly flat layout (`src/`,
`results/`, `docs/`) that didn't match the organizers' required folder names, and
`Documentation_template.md` (with team members' real names) was deliberately kept out of git
history. Both of those choices were reversed so a teammate can clone this repo and produce a
valid submission zip without rebuilding anything by hand. If you want the old flat layout for
reference, it's in the git history before this change.
