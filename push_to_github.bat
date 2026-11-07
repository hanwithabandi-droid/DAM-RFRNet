@echo off
title Push DAM-RFRNet to GitHub
cd /d "%~dp0"

set "GIT_CMD="
if exist "C:\Users\hanwi\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe" (
    set "GIT_CMD=C:\Users\hanwi\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
) else (
    where git >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "GIT_CMD=git"
    )
)

if "%GIT_CMD%"=="" (
    echo Git executable not found.
    pause
    exit /b 1
)

set "MSG=%~1"
if "%MSG%"=="" set "MSG=update project files and documentation"

echo Staging changes...
"%GIT_CMD%" add .

echo Committing: "%MSG%"...
"%GIT_CMD%" commit -m "%MSG%"

echo Pushing to GitHub...
"%GIT_CMD%" push -u origin main

echo.
echo Push completed.
pause
