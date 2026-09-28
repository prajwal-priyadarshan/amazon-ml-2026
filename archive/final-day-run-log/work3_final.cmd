@echo off
cd /d "%~dp0code\business_entity_resolution"
set A=--data-dir ..\..\student_resource\dataset --work-dir ..\..\work3
set P=.venv\Scripts\python.exe -m src.v3.run
%P% ce --split hold3 %A% --ce-out ce2 || exit /b 1
%P% refine-feats --split hold3 %A% --refine-out scored_r10 --use-ce2 || exit /b 1
copy /Y ..\..\work3\refine\holdout_India_scored_r9.parquet ..\..\work3\refine\holdout_India_scored_r10.parquet
copy /Y ..\..\work3\refine\holdout_US_scored_r9.parquet ..\..\work3\refine\holdout_US_scored_r10.parquet
copy /Y ..\..\work3\refine\hold2_India_scored_r9.parquet ..\..\work3\refine\hold2_India_scored_r10.parquet
copy /Y ..\..\work3\refine\hold2_US_scored_r9.parquet ..\..\work3\refine\hold2_US_scored_r10.parquet
copy /Y ..\..\work3\refine\test_France_scored_r9b.parquet ..\..\work3\refine\test_France_scored_r10.parquet
copy /Y ..\..\work3\refine\test_India_scored_r9b.parquet ..\..\work3\refine\test_India_scored_r10.parquet
copy /Y ..\..\work3\refine\test_US_scored_r9b.parquet ..\..\work3\refine\test_US_scored_r10.parquet
%P% train-refine --split holdout %A% --refine-out scored_r10 --refine-splits holdout,hold2,hold3 --refine-leaves 127 --refine-lr 0.025 --refine-bags 3 || exit /b 1
copy /Y ..\..\work3\models\decision_scored_r5.json ..\..\work3\models\decision_scored_r10.json
%P% refine --split test %A% --refine-out scored_r10 || exit /b 1
%P% resolve --split test %A% --scored-dir scored_r10 --out ..\..\output9 || exit /b 1
echo FINAL DONE
