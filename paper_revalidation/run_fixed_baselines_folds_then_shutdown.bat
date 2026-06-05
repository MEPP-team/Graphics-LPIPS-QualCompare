@echo off
setlocal EnableExtensions

if not defined QUALCOMPARE_OUT_ROOT set "QUALCOMPARE_OUT_ROOT=D:\Documents\QualCompare\out"

call "%~dp0revalidate_fixed_baselines_qualcompare.bat" --use-folds --preset BASICS_SP960_YF03_8VP
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto shutdown

call "%~dp0revalidate_fixed_baselines_qualcompare.bat" --use-folds --preset WPC_SP960_YF03_8VP
set "EXIT_CODE=%ERRORLEVEL%"

:shutdown
echo.
echo [INFO] Fold-based fixed-baseline revalidation finished with exit code %EXIT_CODE%.
echo [INFO] Windows will shut down in 60 seconds. Run shutdown /a to cancel.
shutdown /s /t 60 /c "Fold-based fixed-baseline revalidation finished."
exit /b %EXIT_CODE%
