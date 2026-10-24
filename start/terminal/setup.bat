@echo off
REM ===========================================================================
REM GARGANTUA v1 — start/terminal/setup.bat：Windows 构建隔离 Python 运行时
REM ---------------------------------------------------------------------------
REM 与 setup.sh 同职责（详细说明见 setup.sh 文件头），要点：
REM   1. 发现「3.12 系列最新版」python-build-standalone 资产：
REM      curl 拉 atom feed（~15KB）→ findstr 取最新 tag（失败回退内置 tag）；
REM      curl HEAD 探测补 patch 版本；
REM      GARGANTUA_PY_RELEASE / GARGANTUA_PY_VERSION 可手动指定；
REM   2. 下载镜像链：gh-proxy.com → GitHub 直连 → SJTU 镜像；
REM   3. 解包到 start\.python\python\，建 start\.venv\；
REM   4. 装依赖：core.txt 必装；window.txt / dev.txt 按开关；
REM   5. stamp 存在即视为就绪（依赖清单变化时删 start\.venv 强制重建）。
REM 需要 Windows 10+（自带 curl.exe / tar.exe）。
REM
REM 开关（环境变量）：
REM   set GARGANTUA_NO_WINDOW=1   跳过 pywebview 窗口依赖
REM   set GARGANTUA_INSTALL_DEV=1 额外装测试依赖（pytest）
REM   set GARGANTUA_PY_RELEASE=20261021  指定 release tag
REM   set GARGANTUA_PY_VERSION=3.12.15   指定 patch 版本
REM   set GARGANTUA_PY_BASE_URL=...      覆盖下载基址（默认走镜像链）
REM   set PIP_INDEX_URL=...              自定义 pip 源（默认清华）
REM ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
set "PYTHON_DIR=%ROOT%\start\.python"
set "VENV=%ROOT%\start\.venv"
set "REQ_DIR=%ROOT%\start\requirements"
set "STAMP=%VENV%\.gargantua_setup_stamp"
set "PY_SERIES=3.12"
set "TRIPLE=x86_64-pc-windows-msvc"
set "PBS_REPO=astral-sh/python-build-standalone"
set "PBS_FALLBACK_TAG=20261021"
set "TAG_PREFIX=https://github.com/%PBS_REPO%/releases/tag/"

REM ---- 幂等：venv + stamp 存在 → 跳过 ----
if exist "%VENV%\Scripts\python.exe" if exist "%STAMP%" (
  echo [setup] 隔离环境已就绪（%VENV%），跳过安装。
  goto :eof
)

REM ---- 1) 发现 release tag（atom feed；失败回退内置 tag）----
set "TAG=%GARGANTUA_PY_RELEASE%"
if "%TAG%"=="" (
  curl -fsSL --max-time 25 --retry 2 "https://github.com/%PBS_REPO%/releases.atom" -o "%TEMP%\gargantua_atom.xml" 2>nul
  if exist "%TEMP%\gargantua_atom.xml" (
    for /f "delims=" %%l in ('findstr /c:"%TAG_PREFIX%" "%TEMP%\gargantua_atom.xml"') do (
      if not defined TAG (
        set "LINE=%%l"
        REM 去掉 URL 前缀，剩 20261021"/>；tag 为 8 位日期串
        set "TAG=!LINE:*%TAG_PREFIX%=!"
        set "TAG=!TAG:~0,8!"
      )
    )
    del "%TEMP%\gargantua_atom.xml" 2>nul
  )
  REM 校验：8 位数字才算有效（否则回退内置 tag）
  echo !TAG!| findstr /r "^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]$" >nul 2>&1 || set "TAG="
)
if "%TAG%"=="" (
  set "TAG=%PBS_FALLBACK_TAG%"
  echo [setup] atom feed 不可达，使用内置回退 tag=%PBS_FALLBACK_TAG%
) else (
  echo [setup] 最新 release tag=%TAG%
)

REM ---- 2) 探测 3.12.x patch 版本（HEAD 未命中很快，命中一次即停）----
set "VERSION=%GARGANTUA_PY_VERSION%"
if "%VERSION%"=="" (
  for /l %%m in (8,1,30) do (
    if "!VERSION!"=="" (
      set "PROBE_URL=https://github.com/%PBS_REPO%/releases/download/%TAG%/cpython-%PY_SERIES%.%%m+%TAG%-%TRIPLE%-install_only.tar.gz"
      for /f "delims=" %%c in ('curl -s -o NUL -w "%%{http_code}" -I --max-time 15 "!PROBE_URL!" 2^>nul') do set "CODE=%%c"
      if "!CODE!"=="200" set "VERSION=%PY_SERIES%.%%m"
      if "!CODE!"=="302" set "VERSION=%PY_SERIES%.%%m"
    )
  )
)
if "%VERSION%"=="" (
  echo [setup] X tag=%TAG% 内未找到 cpython-3.12.x 资产。
  echo [setup]   可设 GARGANTUA_PY_VERSION=3.12.N 或 GARGANTUA_PY_RELEASE=<tag> 手动指定。
  exit /b 1
)
echo [setup] Python %VERSION% (%TRIPLE%)

REM ---- 3) 下载（镜像链 + 断点续传）----
set "ASSET=cpython-%VERSION%+%TAG%-%TRIPLE%-install_only.tar.gz"
if not exist "%PYTHON_DIR%\python\python.exe" (
  if exist "%TEMP%\gargantua_py.tar.gz" del "%TEMP%\gargantua_py.tar.gz" 2>nul
  set "DL_OK=0"
  if not "%GARGANTUA_PY_BASE_URL%"=="" (
    set "MIRRORS=%GARGANTUA_PY_BASE_URL%"
  ) else (
    set "MIRRORS=https://gh-proxy.com/https://github.com/%PBS_REPO%/releases/download/%TAG%/%ASSET% https://github.com/%PBS_REPO%/releases/download/%TAG%/%ASSET% https://mirror.sjtu.edu.cn/github-release/%PBS_REPO%/releases/download/%TAG%/%ASSET%"
  )
  for %%m in (!MIRRORS!) do (
    if "!DL_OK!"=="0" (
      echo [setup] 下载 %%m
      curl -fSL --retry 3 --retry-delay 2 -C - --max-time 900 "%%m" -o "%TEMP%\gargantua_py.tar.gz"
      if errorlevel 1 (
        echo [setup] 该源失败，尝试下一个……
        del "%TEMP%\gargantua_py.tar.gz" 2>nul
      ) else (
        tar -tzf "%TEMP%\gargantua_py.tar.gz" >nul 2>&1
        if errorlevel 1 (
          echo [setup] 压缩包不完整，尝试下一个……
          del "%TEMP%\gargantua_py.tar.gz" 2>nul
        ) else (
          set "DL_OK=1"
        )
      )
    )
  )
  if "!DL_OK!"=="0" (
    echo [setup] X 下载失败。请检查网络/代理，或设 GARGANTUA_PY_BASE_URL 指向可达镜像。
    exit /b 1
  )
  if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
  tar -xzf "%TEMP%\gargantua_py.tar.gz" -C "%PYTHON_DIR%"
  del "%TEMP%\gargantua_py.tar.gz" 2>nul
  if not exist "%PYTHON_DIR%\python\python.exe" (
    echo [setup] X 解包后未找到 python.exe，安装失败。
    exit /b 1
  )
)

REM ---- 4) 建 venv ----
if not exist "%VENV%\Scripts\python.exe" (
  echo [setup] 创建隔离虚拟环境：%VENV%
  "%PYTHON_DIR%\python\python.exe" -m venv "%VENV%"
)

REM ---- 5) 装依赖 ----
set "PY=%VENV%\Scripts\python.exe"
if "%PIP_INDEX_URL%"=="" (
  set "INDEX_ARGS=-i https://pypi.tuna.tsinghua.edu.cn/simple"
) else (
  set "INDEX_ARGS="
)
"%PY%" -m pip install --quiet --upgrade pip %INDEX_ARGS%
echo [setup] 安装核心依赖（core.txt）...
"%PY%" -m pip install --quiet %INDEX_ARGS% -r "%REQ_DIR%\core.txt"

if not "%GARGANTUA_NO_WINDOW%"=="1" (
  echo [setup] 安装窗口依赖（window.txt，失败不致命）...
  "%PY%" -m pip install --quiet %INDEX_ARGS% -r "%REQ_DIR%\window.txt" || echo [setup] 窗口依赖安装失败（可手动装）
)
if "%GARGANTUA_INSTALL_DEV%"=="1" (
  echo [setup] 安装开发依赖（dev.txt）...
  "%PY%" -m pip install --quiet %INDEX_ARGS% -r "%REQ_DIR%\dev.txt"
)

REM ---- 6) stamp ----
echo done > "%STAMP%"
echo [setup] 隔离环境就绪：%VENV%
echo [setup]   启动：start\terminal\start.bat
endlocal
