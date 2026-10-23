@echo off
setlocal

cd /d "%~dp0"

REM ============================================================
REM GARGANTUA - Windows launcher (thin delegate)
REM
REM This file only locates the system Python. All logic (venv
REM creation, dependency check, Python version guard, server +
REM independent window lifecycle) lives in run.py.
REM
REM 2026-10-23: this file used to carry its own venv/pip logic,
REM which duplicated run.py and drifted from it. Now it only
REM finds an interpreter and execs run.py, same as run.sh /
REM run.command do.
REM ============================================================

set "SYS_PY="

REM Prefer the py launcher (picks the newest 3.x), then python
where py >nul 2>nul
if not errorlevel 1 (
    set "SYS_PY=py -3"
    goto :found
)

where python >nul 2>nul
if not errorlevel 1 (
    set "SYS_PY=python"
    goto :found
)

echo.
echo [GARGANTUA] No system Python found.
echo Please install Python 3.12 ^(torch==2.2.2 has no wheel for 3.13^):
echo   https://www.python.org/downloads/windows/
echo.
pause
exit /b 1

:found
echo [GARGANTUA] Using system Python: %SYS_PY%
echo [GARGANTUA] First launch creates .venv and installs dependencies.
echo.

%SYS_PY% "%~dp0run.py" %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [GARGANTUA] Exited with code %EXITCODE%
    pause
)

endlocal
exit /b %EXITCODE%
