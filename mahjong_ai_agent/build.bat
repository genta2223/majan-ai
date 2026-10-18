@echo off
chcp 65001 > nul
echo ========================================================
echo   Mahjong AI Assistant - Standalone EXE Build Tool
echo ========================================================
echo.

:: Check Python availability
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10+ and enable "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

:: Ensure PyInstaller is installed
echo [1/3] Checking PyInstaller...
python -m pip install --quiet --upgrade pyinstaller
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade PyInstaller via pip. Attempting to continue...
)

:: Run build_exe.py
echo [2/3] Building standalone executable (MahjongAI_Assistant.exe)...
python build_exe.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyInstaller build failed! Please check the logs above.
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo [3/3] Build completed successfully!
echo Executable location: dist\MahjongAI_Assistant.exe
echo ========================================================
echo.
pause
