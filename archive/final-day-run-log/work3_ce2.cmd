@echo off
cd /d "%~dp0code\business_entity_resolution"
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -m src.v3.run train-ce --split fit --data-dir ..\..\student_resource\dataset --work-dir ..\..\work3 --ce-out ce2 --ce-init ce --seed 1 --ce-lr 1e-5 --ce-max-rows 700000 || exit /b 1
echo CE2 TRAINED
