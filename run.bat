@echo off
setlocal

cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

REM ============================================================
REM GARGANTUA - Windows launcher
REM 自动使用系统 Python 创建 .venv
REM ============================================================

if not exist "%PY%" (

    echo [GARGANTUA] 未找到项目 Python 环境，正在寻找系统 Python...

    REM 优先使用 py launcher
    where py >nul 2>nul
    if not errorlevel 1 (
        set "SYS_PY=py -3"
        goto :create_venv
    )

    REM 其次使用 python
    where python >nul 2>nul
    if not errorlevel 1 (
        set "SYS_PY=python"
        goto :create_venv
    )

    echo.
    echo [GARGANTUA] 找不到系统 Python
    echo 请先安装 Python 3.10+
    echo.
    pause
    exit /b 1
)

goto :run


:create_venv

echo [GARGANTUA] 使用系统 Python: %SYS_PY%
echo [GARGANTUA] 正在创建 .venv...

%SYS_PY% -m venv "%VENV%"

if errorlevel 1 (
    echo.
    echo [GARGANTUA] 创建 venv 失败
    echo.
    pause
    exit /b 1
)

if not exist "%PY%" (
    echo.
    echo [GARGANTUA] .venv 创建失败
    echo.
    pause
    exit /b 1
)

REM 安装依赖
if exist requirements.txt (
    echo.
    echo [GARGANTUA] 正在安装依赖...

    "%PY%" -m pip install -U pip

    if errorlevel 1 (
        echo [GARGANTUA] pip 更新失败
        pause
        exit /b 1
    )

    "%PY%" -m pip install -r requirements.txt

    if errorlevel 1 (
        echo [GARGANTUA] 依赖安装失败
        pause
        exit /b 1
    )
)

goto :run


:run

"%PY%" "%~dp0run.py" %*

set "EXITCODE=%ERRORLEVEL%"

endlocal
exit /b %EXITCODE%
