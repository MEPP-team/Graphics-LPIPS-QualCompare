@echo off
setlocal EnableExtensions

rem Run the fixed-baseline revalidation and shut down Windows when it finishes.
rem Pass any optional arguments through to revalidate_fixed_baselines_qualcompare.bat.

call "%~dp0revalidate_fixed_baselines_qualcompare.bat" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [INFO] Revalidation finished with exit code %EXIT_CODE%.
echo [INFO] Windows will shut down in 60 seconds. Press Ctrl+C now to cancel this script,
echo [INFO] or run: shutdown /a

shutdown /s /t 60 /c "Graphics-LPIPS fixed baseline revalidation finished."
exit /b %EXIT_CODE%
