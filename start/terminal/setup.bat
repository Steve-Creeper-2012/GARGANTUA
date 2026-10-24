@echo off
REM ===========================================================================
REM GARGANTUA v1 — start/terminal/setup.bat：Windows 构建隔离 Python 运行时
REM ---------------------------------------------------------------------------
REM 与 setup.sh 同职责（见 setup.sh 文件头），用 PowerShell 拉取
REM python-build-standalone 的最新 3.12.x（Windows 仅 x86_64 官方轮子）。
REM 需要 Windows 10+（自带 curl.exe / tar.exe / PowerShell 5.1）。
REM
REM 开关（环境变量）：
REM   set GARGANTUA_NO_WINDOW=1   跳过 pywebview 窗口依赖
REM   set GARGANTUA_INSTALL_DEV=1 额外装测试依赖（pytest）
REM ===========================================================================
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\.."
set "PYTHON_DIR=%ROOT%\start\.python"
set "VENV=%ROOT%\start\.venv"
set "REQ_DIR=%ROOT%\start\requirements"
set "STAMP=%VENV%\.gargantua_setup_stamp"
set "PY_SERIES=3.12"

REM ---- 幂等：venv 存在且 stamp 存在 → 跳过 ----
if exist "%VENV%\Scripts\python.exe" if exist "%STAMP%" (
  echo [setup] 隔离环境已就绪，跳过安装。
  goto :eof
)

REM ---- 确保下载的独立 Python ----
if not exist "%PYTHON_DIR%\python\python.exe" (
  echo [setup] 下载 python-build-standalone %PY_SERIES%.x ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$api='https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest';" ^
    "$rel=Invoke-RestMethod -Uri $api;" ^
    "$tag=$rel.tag_name;" ^
    "$asset=($rel.assets | Where-Object { $_.name -like 'cpython-%PY_SERIES%.*-x86_64-pc-windows-msvc-shared-install_only.tar.gz' } | Select-Object -First 1);" ^
    "if(-not $asset){Write-Error 'no matching asset';exit 1};" ^
    "$url=$asset.browser_download_url;" ^
    "Write-Host ('tag='+$tag);" ^
    "Invoke-WebRequest -Uri $url -OutFile $env:TEMP\gargantua_py.tar.gz;"
  if errorlevel 1 (
    echo [setup] X 下载失败，请检查网络/代理，或手动下载 python-build-standalone 3.12.x 解压到 %PYTHON_DIR%\python\
    exit /b 1
  )
  if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
  tar -xzf "%TEMP%\gargantua_py.tar.gz" -C "%PYTHON_DIR%"
  del "%TEMP%\gargantua_py.tar.gz"
)

REM ---- 建 venv ----
if not exist "%VENV%\Scripts\python.exe" (
  echo [setup] 创建隔离虚拟环境：%VENV%
  "%PYTHON_DIR%\python\python.exe" -m venv "%VENV%"
)

REM ---- 装依赖 ----
set "PY=%VENV%\Scripts\python.exe"
"%PY%" -m pip install --quiet --upgrade pip
echo [setup] 安装核心依赖（core.txt）...
"%PY%" -m pip install --quiet -r "%REQ_DIR%\core.txt"

if not "%GARGANTUA_NO_WINDOW%"=="1" (
  echo [setup] 安装窗口依赖（window.txt，失败不致命）...
  "%PY%" -m pip install --quiet -r "%REQ_DIR%\window.txt" || echo [setup] 窗口依赖安装失败（可手动装）
)
if "%GARGANTUA_INSTALL_DEV%"=="1" (
  echo [setup] 安装开发依赖（dev.txt）...
  "%PY%" -m pip install --quiet -r "%REQ_DIR%\dev.txt"
)

echo done > "%STAMP%"
echo [setup] 隔离环境就绪：%VENV%
echo [setup]   启动：start\terminal\start.bat
endlocal
