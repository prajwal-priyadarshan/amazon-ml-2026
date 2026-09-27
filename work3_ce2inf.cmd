@echo off
cd /d "%~dp0code\business_entity_resolution"
set A=--data-dir ..\..\student_resource\dataset --work-dir ..\..\work3
set P=.venv\Scripts\python.exe -m src.v3.run
%P% ce --split holdout %A% --ce-out ce2 || exit /b 1
%P% ce --split hold2 %A% --ce-out ce2 || exit /b 1
%P% ce --split test %A% --ce-out ce2 || exit /b 1
echo CE2 INFER DONE
