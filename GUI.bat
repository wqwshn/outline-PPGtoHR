@echo off
setlocal
cd /d "%~dp0"
title PPG-HR Paper Replay

set "REPLAY_CONDA="
for /f "delims=" %%I in ('where conda.bat 2^>nul') do if not defined REPLAY_CONDA set "REPLAY_CONDA=%%I"
if not defined REPLAY_CONDA if defined CONDA_EXE if exist "%CONDA_EXE%" set "REPLAY_CONDA=%CONDA_EXE%"
if not defined REPLAY_CONDA for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined REPLAY_CONDA set "REPLAY_CONDA=%%I"
if not defined REPLAY_CONDA for %%I in ("D:\Anaconda\condabin\conda.bat" "%USERPROFILE%\anaconda3\condabin\conda.bat" "%USERPROFILE%\miniconda3\condabin\conda.bat" "%ProgramData%\anaconda3\condabin\conda.bat" "%ProgramData%\miniconda3\condabin\conda.bat") do if not defined REPLAY_CONDA if exist "%%~I" set "REPLAY_CONDA=%%~I"

if not defined REPLAY_CONDA (
    echo Conda was not found. Install Conda and create the ppg-hr environment.
    echo You can also run this file from Anaconda Prompt.
    pause
    exit /b 1
)

echo Starting the paper replay GUI with the ppg-hr environment...
call "%REPLAY_CONDA%" run --no-capture-output -n ppg-hr python "%~dp0tools\launch_paper_replay.py"
set "REPLAY_EXIT=%ERRORLEVEL%"
if not "%REPLAY_EXIT%"=="0" (
    echo.
    echo GUI startup failed with exit code %REPLAY_EXIT%. See the error above.
    pause
)
exit /b %REPLAY_EXIT%
