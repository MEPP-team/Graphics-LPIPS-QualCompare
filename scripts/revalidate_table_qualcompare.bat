@echo off
setlocal EnableDelayedExpansion

rem -----------------------------------------------------------------------------
rem Revalidate paper table entries (new Graphics-LPIPS rows) with QualCompare data
rem -----------------------------------------------------------------------------
rem This script does NOT render data. Renders must be prepared first with QualCompare.
rem
rem Expected render tree under SRC_ROOT:
rem   Source\<N>VP\<REFERENCE_OBJ>\views\view_1.png ...
rem   Source\<N>VP\<REFERENCE_OBJ>\patchs\<REFERENCE_OBJ>_patchlist.csv
rem   Distorted\<N>VP\<DISTORTED_OBJ>\views\view_1.png ...
rem
rem Pipeline reproduced from old config.py behavior:
rem   1) train.py                    -> train the model and create checkpoints
rem   2) Light_GraphicsLPIPS_csv.py  -> compute per-object metric CSVs
rem   3) correlation_VP.py           -> compute PLCC/SROCC summary
rem -----------------------------------------------------------------------------

set "PYTHON=python"
set "REPO_ROOT=%~dp0.."

rem Set to 1 to print commands only.
set "DRY_RUN=0"
if /I "%~1"=="--dry-run" set "DRY_RUN=1"

rem =========================
rem STEP 0 - SELECT A PRESET
rem =========================
rem Keep ONLY one preset active.

rem --- PRESET A: TMQ, 5-fold (table row: Graphics-LPIPS, 5-fold)
set "RUN_NAME=TMQ_5FOLD"
set "MODEL_NAME=TMQ_NR_8VP_yf03_kfolds"
set "DATABASE=TMQ"
set "VIEWS=8"
set "VIEW_METHOD=Y_fixed_0.3"
set "RENDER_METHOD=New_Render"
set "MOS_CSV=.\dataset\TMQ\TMQ_MOS.csv"
set "TESTLIST_CSV=.\dataset\TMQ\folds\TexturedDB_20_TestList_withnbPatchesPerVP_threth0.6.csv"
set "USE_FOLDS=1"

rem --- PRESET B: TSMD zero-shot (example; update paths/csv with your local files)
rem set "RUN_NAME=TSMD_ZEROSHOT"
rem set "MODEL_NAME=TMQ_NR_8VP_yf03_kfolds"  rem can use the same model as TMQ since it's zero-shot
rem set "DATABASE=TSMD"
rem set "VIEWS=8"
rem set "VIEW_METHOD=Y_fixed_0"
rem set "RENDER_METHOD=New_Render"
rem set "MOS_CSV=.\dataset\TSMD\TSMD_MOS.csv"
rem set "TESTLIST_CSV=.\dataset\TSMD\folds\TSMD_test20.csv"
rem set "USE_FOLDS=0"

rem --- PRESET C: SJTU-TMQA zero-shot (example; update paths/csv with your local files)
rem set "RUN_NAME=SJTU_TMQA_ZEROSHOT"
rem set "MODEL_NAME=TMQ_NR_8VP_yf03_kfolds"  rem can use the same model as TMQ since it's zero-shot
rem set "DATABASE=SJTU-TMQA"
rem set "VIEWS=8"
rem set "VIEW_METHOD=Y_fixed_0"
rem set "RENDER_METHOD=0_0_light"
rem set "MOS_CSV=.\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv"
rem set "TESTLIST_CSV=.\dataset\SJTU-TMQA\SJTU-TMQA_MOS_normalized.csv"
rem set "USE_FOLDS=0"

rem --- PRESET D: BASICS zero-shot (example; update paths/csv with your local files)
rem set "RUN_NAME=BASICS_ZEROSHOT"
rem set "MODEL_NAME=TMQ_NR_8VP_yf03_kfolds"  rem can use the same model as TMQ since it's zero-shot
rem set "DATABASE=BASICS"
rem set "VIEWS=8"
rem set "VIEW_METHOD=Y_fixed_0"
rem set "RENDER_METHOD=SP"
rem set "MOS_CSV=.\dataset\BASICS\MOS_CI.csv"
rem set "TESTLIST_CSV=.\dataset\BASICS\folds\MOS_CI_test20.csv"
rem set "USE_FOLDS=0"

rem =========================
rem STEP 1 - USER PARAMETERS
rem =========================
rem IMPORTANT: set this to your QualCompare render output root for the selected preset.
rem Example: D:\RenderedDatasets\TMQ\New_Render\Y_fixed_0.3
set "SRC_ROOT=D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3"

rem Output root used by Light_GraphicsLPIPS_csv.py and correlation_VP.py
set "OUT_ROOT=%REPO_ROOT%\out"

rem Set USE_GPU=0 if CUDA is unavailable.
set "USE_GPU=1"

rem Train before validation for the active preset.
set "RUN_TRAINING=1"

rem Keep only latest_net_.pth after training. Set to 0 if you want epoch snapshots.
set "KEEP_ONLY_LATEST=1"

rem Training hyperparameters for the active preset.
set "TRAIN_NAME=%MODEL_NAME%"
set "TRAIN_DATASETS=%REPO_ROOT%\dataset\TMQ\folds\TexturedDB_80_TrainList_withnbPatchesPerVP_threth0.6.csv"
set "TRAIN_TESTCSV=%REPO_ROOT%\dataset\TMQ\folds\TexturedDB_20_TestList_withnbPatchesPerVP_threth0.6.csv"
set "TRAIN_TARGET=judges"
set "TRAIN_CHECKPOINTS_DIR=%REPO_ROOT%\checkpoints"
set "TRAIN_ROOT_REFPATCHES=Source\%VIEWS%VP"
set "TRAIN_ROOT_DISTPATCHES=Distorted\%VIEWS%VP"
set "TRAIN_NET=alex"
set "TRAIN_NPATCHES=150"
set "TRAIN_NINPUTIMG=4"
set "TRAIN_NEPOCH=5"
set "TRAIN_NEPOCH_DECAY=5"
set "TRAIN_SAVE_EPOCH_FREQ=10"
set "TRAIN_NTHREADS=8"
set "TRAIN_FOLDS_FLAG="
if "%USE_FOLDS%"=="1" set "TRAIN_FOLDS_FLAG=--use_folds"

rem =========================
rem STEP 2 - BASIC CHECKS
rem =========================
if "%SRC_ROOT%"=="" (
  echo [ERROR] SRC_ROOT is empty.
  echo [INFO ] Generate renders first with QualCompare, then set SRC_ROOT in this file.
  echo [INFO ] Needed folders under SRC_ROOT: Source\%VIEWS%VP and Distorted\%VIEWS%VP
  exit /b 1
)

if not exist "%SRC_ROOT%\Source\%VIEWS%VP" (
  echo [ERROR] Missing folder: %SRC_ROOT%\Source\%VIEWS%VP
  exit /b 1
)

if not exist "%SRC_ROOT%\Distorted\%VIEWS%VP" (
  echo [ERROR] Missing folder: %SRC_ROOT%\Distorted\%VIEWS%VP
  exit /b 1
)

if not exist "%REPO_ROOT%\%MOS_CSV%" (
  echo [ERROR] Missing MOS CSV: %REPO_ROOT%\%MOS_CSV%
  exit /b 1
)

if "%USE_FOLDS%"=="1" (
  set "TRAIN_DATASETS_K0=!TRAIN_DATASETS:.csv=_k0.csv!"
  if not exist "!TRAIN_DATASETS_K0!" (
    echo [ERROR] Missing training fold CSV: !TRAIN_DATASETS_K0!
    exit /b 1
  )

  set "TRAIN_TESTCSV_K0=!TRAIN_TESTCSV:.csv=_k0.csv!"
  if not exist "!TRAIN_TESTCSV_K0!" (
    echo [ERROR] Missing training test fold CSV: !TRAIN_TESTCSV_K0!
    exit /b 1
  )
) else (
  if not exist "%TRAIN_DATASETS%" (
    echo [ERROR] Missing training CSV: %TRAIN_DATASETS%
    exit /b 1
  )

  if not exist "%TRAIN_TESTCSV%" (
    echo [ERROR] Missing training test CSV: %TRAIN_TESTCSV%
    exit /b 1
  )
)

if "%USE_FOLDS%"=="1" (
  set "TESTLIST_BASE=%REPO_ROOT%\!TESTLIST_CSV!"
  set "TESTLIST_K0=!TESTLIST_BASE:.csv=_k0.csv!"
  if not exist "!TESTLIST_K0!" (
    echo [ERROR] Missing fold test list CSV: !TESTLIST_K0!
    exit /b 1
  )
) else (
  if not exist "%REPO_ROOT%\%TESTLIST_CSV%" (
    echo [ERROR] Missing test list CSV: %REPO_ROOT%\%TESTLIST_CSV%
    exit /b 1
  )
)

pushd "%REPO_ROOT%"

if "%DRY_RUN%"=="0" (
  if "%RUN_TRAINING%"=="0" (
    if "%USE_FOLDS%"=="1" (
      if not exist "%REPO_ROOT%\checkpoints\%MODEL_NAME%\fold_k0\latest_net_.pth" (
        echo [ERROR] Fold checkpoint not found:
        echo         %REPO_ROOT%\checkpoints\%MODEL_NAME%\fold_k0\latest_net_.pth
        exit /b 1
      )
    ) else (
      if not exist "%REPO_ROOT%\checkpoints\%MODEL_NAME%\latest_net_.pth" (
        echo [ERROR] Checkpoint not found:
        echo         %REPO_ROOT%\checkpoints\%MODEL_NAME%\latest_net_.pth
        exit /b 1
      )
    )
  )
)

if "%RUN_TRAINING%"=="1" (
  echo [STEP 3/5] Training
  echo %PYTHON% train.py --name %TRAIN_NAME% %TRAIN_FOLDS_FLAG% --src_root "%SRC_ROOT%" --datasets "%TRAIN_DATASETS%" --testcsv "%TRAIN_TESTCSV%" --root_refPatches %TRAIN_ROOT_REFPATCHES% --root_distPatches %TRAIN_ROOT_DISTPATCHES% --target %TRAIN_TARGET% --net %TRAIN_NET% --npatches %TRAIN_NPATCHES% --nInputImg %TRAIN_NINPUTIMG% --nepoch %TRAIN_NEPOCH% --nepoch_decay %TRAIN_NEPOCH_DECAY% --save_epoch_freq %TRAIN_SAVE_EPOCH_FREQ% --nThreads %TRAIN_NTHREADS% --checkpoints_dir "%TRAIN_CHECKPOINTS_DIR%"
  if "%DRY_RUN%"=="0" (
    %PYTHON% train.py --name %TRAIN_NAME% %TRAIN_FOLDS_FLAG% --src_root "%SRC_ROOT%" --datasets "%TRAIN_DATASETS%" --testcsv "%TRAIN_TESTCSV%" --root_refPatches %TRAIN_ROOT_REFPATCHES% --root_distPatches %TRAIN_ROOT_DISTPATCHES% --target %TRAIN_TARGET% --net %TRAIN_NET% --npatches %TRAIN_NPATCHES% --nInputImg %TRAIN_NINPUTIMG% --nepoch %TRAIN_NEPOCH% --nepoch_decay %TRAIN_NEPOCH_DECAY% --save_epoch_freq %TRAIN_SAVE_EPOCH_FREQ% --nThreads %TRAIN_NTHREADS% --checkpoints_dir "%TRAIN_CHECKPOINTS_DIR%"
    if errorlevel 1 (
      echo [ERROR] Training failed.
      popd
      exit /b 1
    )
  )

  if "%DRY_RUN%"=="0" if "%KEEP_ONLY_LATEST%"=="1" (
    if "%USE_FOLDS%"=="1" (
      for /d %%D in ("%TRAIN_CHECKPOINTS_DIR%\%TRAIN_NAME%\fold_k*") do call :CleanupCheckpoints "%%~fD"
    ) else (
      call :CleanupCheckpoints "%TRAIN_CHECKPOINTS_DIR%\%TRAIN_NAME%"
    )
  )
)

set "GPU_FLAG="
if "%USE_GPU%"=="1" set "GPU_FLAG=--use_gpu"

set "FOLDS_FLAG="
if "%USE_FOLDS%"=="1" set "FOLDS_FLAG=--use_folds"

echo ============================================================================
echo RUN_NAME     : %RUN_NAME%
echo MODEL_NAME   : %MODEL_NAME%
echo DATABASE     : %DATABASE%
echo VIEWS        : %VIEWS%
echo VIEW_METHOD  : %VIEW_METHOD%
echo RENDER_METHOD: %RENDER_METHOD%
echo SRC_ROOT     : %SRC_ROOT%
echo OUT_ROOT     : %OUT_ROOT%
echo USE_FOLDS    : %USE_FOLDS%
echo USE_GPU      : %USE_GPU%
echo DRY_RUN      : %DRY_RUN%
echo ============================================================================

set "EVAL_CMD=%PYTHON% Light_GraphicsLPIPS_csv.py -m %MODEL_NAME% %FOLDS_FLAG% -v %VIEWS% -vm %VIEW_METHOD% -rm %RENDER_METHOD% -db %DATABASE% -mos %MOS_CSV% -testlist %TESTLIST_CSV% --src_root "%SRC_ROOT%" %GPU_FLAG%"
set "CORR_CMD=%PYTHON% correlation_VP.py -m %MODEL_NAME% %FOLDS_FLAG% -v %VIEWS% -vm %VIEW_METHOD% -rm %RENDER_METHOD% -db %DATABASE% --out_root "%OUT_ROOT%""

echo [STEP 4/5] Metric evaluation
echo %EVAL_CMD%
if "%DRY_RUN%"=="0" (
  %PYTHON% Light_GraphicsLPIPS_csv.py -m %MODEL_NAME% %FOLDS_FLAG% -v %VIEWS% -vm %VIEW_METHOD% -rm %RENDER_METHOD% -db %DATABASE% -mos %MOS_CSV% -testlist %TESTLIST_CSV% --src_root "%SRC_ROOT%" %GPU_FLAG%
  if errorlevel 1 (
    echo [ERROR] Metric evaluation failed.
    popd
    exit /b 1
  )
)

echo [STEP 5/5] Correlation summary
echo %CORR_CMD%
if "%DRY_RUN%"=="0" (
  %PYTHON% correlation_VP.py -m %MODEL_NAME% %FOLDS_FLAG% -v %VIEWS% -vm %VIEW_METHOD% -rm %RENDER_METHOD% -db %DATABASE% --out_root "%OUT_ROOT%"
  if errorlevel 1 (
    echo [ERROR] Correlation failed.
    popd
    exit /b 1
  )
)

set "RESULT_DIR=%OUT_ROOT%\%DATABASE%\%RENDER_METHOD%\%VIEW_METHOD%\%MODEL_NAME%\%VIEWS%VP"
echo [DONE ] Revalidation finished.
echo [OUT  ] Main output directory: %RESULT_DIR%

if "%USE_FOLDS%"=="1" (
  echo [OUT  ] Fold summary: %RESULT_DIR%\correlation_folds_stats.csv
) else (
  echo [OUT  ] Single summary: %RESULT_DIR%\correlation_folds_stats.csv
)

popd
endlocal
exit /b 0

:CleanupCheckpoints
set "CLEAN_DIR=%~1"
if not exist "%CLEAN_DIR%" exit /b 0

for %%F in ("%CLEAN_DIR%\*_net_*.pth") do (
  if exist "%%~fF" (
    if /I not "%%~nxF"=="latest_net_.pth" del /q "%%~fF"
  )
)
exit /b 0
