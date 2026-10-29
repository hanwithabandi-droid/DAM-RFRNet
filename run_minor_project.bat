@echo off
echo =====================================================================
echo       DAM-RFRNet Minor Project: Single Image Neural Dehazing
echo =====================================================================
echo.
echo Launching DAM-RFRNet Single Image Evaluation...
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe manual_infer.py %*
) else (
    python manual_infer.py %*
)

echo.
echo Process complete. Check the ./outputs/ folder for saved results.
pause
