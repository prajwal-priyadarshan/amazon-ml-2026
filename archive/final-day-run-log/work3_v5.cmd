@echo off
cd /d "%~dp0code\business_entity_resolution"
set A=--data-dir ..\..\student_resource\dataset --work-dir ..\..\work3
.venv\Scripts\python.exe -m src.v3.run refine-feats --split test %A% || exit /b 1
.venv\Scripts\python.exe -m src.v3.run refine --split test %A% || exit /b 1
.venv\Scripts\python.exe -m src.v3.run resolve --split test %A% --scored-dir scored_r --out ..\..\output5 || exit /b 1
.venv\Scripts\python.exe ..\..\student_resource\utils\validate_submission.py --matching ..\..\output5\matching_results.tsv --candidate ..\..\output5\candidate_pairs.tsv --test-dir ..\..\student_resource\dataset\test
echo V5 DONE
