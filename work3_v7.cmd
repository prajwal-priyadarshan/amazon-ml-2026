@echo off
cd /d "%~dp0code\business_entity_resolution"
set A=--data-dir ..\..\student_resource\dataset --work-dir ..\..\work3
set P=.venv\Scripts\python.exe -m src.v3.run
echo === output7: refine r8 on test
%P% refine-feats --split test %A% --refine-out scored_r8 || exit /b 1
%P% refine --split test %A% --refine-out scored_r8 || exit /b 1
%P% resolve --split test %A% --scored-dir scored_r8 --out ..\..\output7 || exit /b 1
echo === hold3 pipeline
%P% prep --split hold3 %A% --jobs 6 || exit /b 1
%P% lexical --split hold3 %A% || exit /b 1
%P% embed --split hold3 %A% --emb-tag e5s-ft || exit /b 1
%P% dense --split hold3 %A% --emb-tag e5s-ft || exit /b 1
%P% union --split hold3 %A% --emb-tag e5s-ft || exit /b 1
%P% prune --split hold3 %A% --expand || exit /b 1
%P% ce --split hold3 %A% || exit /b 1
%P% stack --split hold3 %A% --save-feats || exit /b 1
echo HOLD3 DONE
