@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Revalidate fixed baseline metrics on the paper datasets:
rem   - LPIPS_TORCHMETRICS: patch-based LPIPS
rem   - SSIM: patch-based SSIM
rem   - SSIM_IMAGES: full-view SSIM
rem
rem These baselines are not trained, so they are evaluated once on the full
rem rendered database. No folds are used.

set "DRY_RUN=0"
if /I "%~1"=="--dry-run" set "DRY_RUN=1"

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" >nul

set "PYTHON=python"
set "OUT_ROOT=.\out"
set "GPU_FLAG=--use_gpu"

if defined QUALCOMPARE_OUT_ROOT (
  set "RENDERS_ROOT=%QUALCOMPARE_OUT_ROOT%"
) else (
  set "RENDERS_ROOT=D:\These\Projets\CompareMetrics\out"
)

echo [INFO] Render root: %RENDERS_ROOT%
echo [INFO] Dry run    : %DRY_RUN%
echo.

call :RunDataset TMQ 8 New_Render Y_fixed_0.3 ".\dataset\TMQ\TMQ_MOS.csv"
if errorlevel 1 goto :fail

rem Local available TSMD renders use Y_fixed_0.3. Change this line to Y_fixed_0
rem if you regenerate or move the paper renders to that view method.
call :RunDataset TSMD 8 New_Render Y_fixed_0.3 ".\dataset\TSMD\_TSMD_fulldataset.csv"
if errorlevel 1 goto :fail

call :RunDataset SJTU-TMQA 8 0_0_light Y_fixed_0 ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv"
if errorlevel 1 goto :fail

call :RunDataset BASICS 4 SP Y_fixed_0 ".\dataset\BASICS\MOS_CI.csv"
if errorlevel 1 goto :fail

echo.
echo [OK] Fixed baseline revalidation finished.
popd >nul
exit /b 0

:RunDataset
set "DATABASE=%~1"
set "VIEWS=%~2"
set "RENDER_METHOD=%~3"
set "VIEW_METHOD=%~4"
set "MOS_CSV=%~5"
set "SRC_ROOT=%RENDERS_ROOT%\%DATABASE%\%RENDER_METHOD%\%VIEW_METHOD%"

echo ============================================================
echo [DATASET] %DATABASE% / %RENDER_METHOD% / %VIEW_METHOD% / %VIEWS%VP
echo [SRC_ROOT] %SRC_ROOT%

if not exist "%SRC_ROOT%\Source\%VIEWS%VP" (
  echo [ERROR] Missing folder: %SRC_ROOT%\Source\%VIEWS%VP
  exit /b 1
)
if not exist "%SRC_ROOT%\Distorted\%VIEWS%VP" (
  echo [ERROR] Missing folder: %SRC_ROOT%\Distorted\%VIEWS%VP
  exit /b 1
)
if not exist "%MOS_CSV%" (
  echo [ERROR] Missing MOS CSV: %MOS_CSV%
  exit /b 1
)

call :RunMetric revalidate_lpips.py LPIPS_TORCHMETRICS "%DATABASE%" %VIEWS% "%RENDER_METHOD%" "%VIEW_METHOD%" "%MOS_CSV%" "%SRC_ROOT%" %GPU_FLAG%
if errorlevel 1 exit /b 1
call :RunCorrelation LPIPS_TORCHMETRICS "%DATABASE%" %VIEWS% "%RENDER_METHOD%" "%VIEW_METHOD%"
if errorlevel 1 exit /b 1

call :RunMetric revalidate_ssim.py SSIM "%DATABASE%" %VIEWS% "%RENDER_METHOD%" "%VIEW_METHOD%" "%MOS_CSV%" "%SRC_ROOT%"
if errorlevel 1 exit /b 1
call :RunCorrelation SSIM "%DATABASE%" %VIEWS% "%RENDER_METHOD%" "%VIEW_METHOD%"
if errorlevel 1 exit /b 1

call :RunMetric revalidate_ssim_images.py SSIM_IMAGES "%DATABASE%" %VIEWS% "%RENDER_METHOD%" "%VIEW_METHOD%" "%MOS_CSV%" "%SRC_ROOT%"
if errorlevel 1 exit /b 1
call :RunCorrelation SSIM_IMAGES "%DATABASE%" %VIEWS% "%RENDER_METHOD%" "%VIEW_METHOD%"
if errorlevel 1 exit /b 1

exit /b 0

:RunMetric
set "SCRIPT=%~1"
set "MODEL=%~2"
set "DATABASE=%~3"
set "VIEWS=%~4"
set "RENDER_METHOD=%~5"
set "VIEW_METHOD=%~6"
set "MOS_CSV=%~7"
set "SRC_ROOT=%~8"
set "EXTRA_FLAG=%~9"

set "CMD=%PYTHON% %SCRIPT% -m %MODEL% -v %VIEWS% -vm %VIEW_METHOD% -rm %RENDER_METHOD% -db %DATABASE% -mos %MOS_CSV% --src_root "%SRC_ROOT%" %EXTRA_FLAG%"
echo.
echo [METRIC] %MODEL%
echo %CMD%
call :MetricResultsFilename "%MODEL%"
set "RESULT_ROOT=%OUT_ROOT%\%DATABASE%\%RENDER_METHOD%\%VIEW_METHOD%\%MODEL%\%VIEWS%VP\_METRIC_RESULTS_TESTSET_"
call :MetricIsComplete "%SRC_ROOT%\Source\%VIEWS%VP" "%RESULT_ROOT%" "%RESULT_FILENAME%"
if "%METRIC_COMPLETE%"=="1" (
  echo [SKIP] Existing complete results found for %MODEL%: %RESULT_ROOT%
  exit /b 0
)
if "%DRY_RUN%"=="0" (
  %PYTHON% %SCRIPT% -m %MODEL% -v %VIEWS% -vm %VIEW_METHOD% -rm %RENDER_METHOD% -db %DATABASE% -mos %MOS_CSV% --src_root "%SRC_ROOT%" %EXTRA_FLAG%
  if errorlevel 1 exit /b 1
)
exit /b 0

:MetricResultsFilename
set "RESULT_FILENAME=GLPIPS_results_testset.csv"
if /I "%~1"=="LPIPS_TORCHMETRICS" set "RESULT_FILENAME=LPIPS_results_testset.csv"
if /I "%~1"=="SSIM" set "RESULT_FILENAME=SSIM_results_testset.csv"
if /I "%~1"=="SSIM_IMAGES" set "RESULT_FILENAME=SSIM_IMAGES_results_testset.csv"
if /I "%~1"=="WEIGHTED_GLPIPS" set "RESULT_FILENAME=WEIGHTED_GLPIPS_results_testset.csv"
exit /b 0

:MetricIsComplete
set "METRIC_COMPLETE=0"
for /f %%A in ('%PYTHON% -c "import os,sys; src,root,name=sys.argv[1:4]; refs=[d for d in os.listdir(src) if os.path.isdir(os.path.join(src,d))] if os.path.isdir(src) else []; done=sum(1 for r in refs if os.path.isfile(os.path.join(root,r,name)) and len(open(os.path.join(root,r,name), newline='').read().splitlines())>1); print('1' if refs and done>=len(refs) else '0')" "%~1" "%~2" "%~3"') do set "METRIC_COMPLETE=%%A"
exit /b 0

:RunCorrelation
set "MODEL=%~1"
set "DATABASE=%~2"
set "VIEWS=%~3"
set "RENDER_METHOD=%~4"
set "VIEW_METHOD=%~5"

set "CMD=%PYTHON% correlation_VP.py -m %MODEL% -v %VIEWS% -vm %VIEW_METHOD% -rm %RENDER_METHOD% -db %DATABASE% --out_root "%OUT_ROOT%""
echo.
echo [CORRELATION] %MODEL%
echo %CMD%
if "%DRY_RUN%"=="0" (
  %PYTHON% correlation_VP.py -m %MODEL% -v %VIEWS% -vm %VIEW_METHOD% -rm %RENDER_METHOD% -db %DATABASE% --out_root "%OUT_ROOT%"
  if errorlevel 1 exit /b 1
)
exit /b 0

:fail
echo.
echo [FAILED] Fixed baseline revalidation stopped.
popd >nul
exit /b 1
