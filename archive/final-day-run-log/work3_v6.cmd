@echo off
cd /d "%~dp0code\business_entity_resolution"
set A=--data-dir ..\..\student_resource\dataset --work-dir ..\..\work3
set P=.venv\Scripts\python.exe -m src.v3.run




echo === hold2 pipeline

%P% lexical --split hold2 %A% || exit /b 1
%P% embed --split hold2 %A% --emb-tag e5s-ft || exit /b 1
%P% dense --split hold2 %A% --emb-tag e5s-ft || exit /b 1
%P% union --split hold2 %A% --emb-tag e5s-ft || exit /b 1
%P% prune --split hold2 %A% --expand || exit /b 1
%P% ce --split hold2 %A% || exit /b 1
%P% stack --split hold2 %A% --save-feats || exit /b 1
echo HOLD2 DONE
