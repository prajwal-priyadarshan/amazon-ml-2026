@echo off
cd /d "%~dp0code\business_entity_resolution"
.venv\Scripts\python.exe ..\..\student_resource\utils\validate_submission.py --matching ..\..\output6\matching_results.tsv --candidate ..\..\output6\candidate_pairs.tsv --test-dir ..\..\student_resource\dataset\test
echo VAL6 DONE
