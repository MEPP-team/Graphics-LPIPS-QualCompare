@echo off
setlocal EnableExtensions

rem Run the Graphics-LPIPS training/evaluation revalidation and shut down Windows
rem when it finishes. By default, this runs the current paper presets for BASICS
rem and WPC. Pass optional arguments to run a single custom revalidate_table call.

set "SCRIPT_DIR=%~dp0"
set "DRY_RUN=0"

if not defined REVALIDATION_PYTHON set "REVALIDATION_PYTHON=python"
if not defined REVALIDATION_NTHREADS set "REVALIDATION_NTHREADS=0"
if not defined QUALCOMPARE_OUT_ROOT set "QUALCOMPARE_OUT_ROOT=D:\Documents\QualCompare\out"

for %%A in (%*) do (
  if /I "%%~A"=="--dry-run" set "DRY_RUN=1"
  if /I "%%~A"=="-dry-run" set "DRY_RUN=1"
)

if "%~1"=="" goto run_defaults

call "%SCRIPT_DIR%revalidate_table_qualcompare.bat" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto shutdown

:run_defaults
call "%SCRIPT_DIR%revalidate_table_qualcompare.bat" --preset BASICS_SP960_YF03_8VP_5FOLD
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto shutdown

call "%SCRIPT_DIR%revalidate_table_qualcompare.bat" --preset WPC_SP960_YF03_8VP_5FOLD
set "EXIT_CODE=%ERRORLEVEL%"

:shutdown
echo.
echo [INFO] Graphics-LPIPS revalidation finished with exit code %EXIT_CODE%.
if "%DRY_RUN%"=="1" (
  echo [INFO] Dry run detected; Windows shutdown is skipped.
  exit /b %EXIT_CODE%
)
echo [INFO] Windows will shut down in 60 seconds. Press Ctrl+C now to cancel this script,
echo [INFO] or run: shutdown /a

shutdown /s /t 60 /c "Graphics-LPIPS revalidation finished."
exit /b %EXIT_CODE%
