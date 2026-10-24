@echo off
REM ===========================================================================
REM GARGANTUA v1 — start/terminal/setup.bat：Windows 构建隔离 Python 运行时
REM ---------------------------------------------------------------------------
REM 与 setup.sh 同职责（详细说明见 setup.sh 文件头），要点：
REM   1. 发现「3.12 系列最新版」python-build-standalone 资产：
REM      curl 拉 atom feed（~15KB）→ findstr 取最新 tag（失败回退内置 tag）；
REM      curl HEAD 探测补 patch 版本；
REM      GARGANTUA_PY_RELEASE / GARGANTUA_PY_VERSION 可手动指定；
REM   2. 下载镜像【自动测速】：对候选镜像顺序发 Range 请求（前 2MB，curl
REM      自报 speed_download），选最快的下载；失败按默认顺序回退。
REM      GARGANTUA_PY_BASE_URL 可直接指定（跳过测速）；
REM   3. 解包到 environment\python\，建 environment\venv\（含自愈：解释器
REM      跑不了就重建）；
REM   4. 装依赖：core.txt 必装；window.txt / dev.txt 按开关；
REM   5. stamp 存在即视为就绪（依赖清单变化时删 environment\.venv 强制重建）。
REM 需要 Windows 10+（自带 curl.exe / tar.exe）。
REM
REM 开关（环境变量）：
REM   set GARGANTUA_NO_WINDOW=1   跳过 pywebview 窗口依赖
REM   set GARGANTUA_INSTALL_DEV=1 额外装测试依赖（pytest）
REM   set GARGANTUA_PY_RELEASE=20261021  指定 release tag
REM   set GARGANTUA_PY_VERSION=3.12.15   指定 patch 版本
REM   set GARGANTUA_PY_BASE_URL=...      指定下载镜像（跳过测速）
REM   set PIP_INDEX_URL=...              自定义 pip 源（跳过测速）
REM ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
set "PYTHON_DIR=%ROOT%\environment\python"
set "VENV=%ROOT%\environment\venv"
set "REQ_DIR=%ROOT%\start\requirements"
set "STAMP=%VENV%\.gargantua_setup_stamp"
set "PY_SERIES=3.12"
set "TRIPLE=x86_64-pc-windows-msvc"
set "PBS_REPO=astral-sh/python-build-standalone"
set "PBS_FALLBACK_TAG=20261021"
set "TAG_PREFIX=https://github.com/%PBS_REPO%/releases/tag/"

REM ---- 幂等：venv + stamp 存在且解释器能跑 → 跳过 ----
if exist "%VENV%\Scripts\python.exe" if exist "%STAMP%" (
  "%VENV%\Scripts\python.exe" -c "" >nul 2>&1
  if not errorlevel 1 (
    echo [setup] 隔离环境已就绪（%VENV%），跳过安装。
    goto :eof
  )
  echo [setup] environment\venv 不可用（项目目录被移动过？），重建……
  rmdir /s /q "%VENV%"
)

REM ---- 1) 发现 release tag（atom feed；失败回退内置 tag）----
set "TAG=%GARGANTUA_PY_RELEASE%"
if "%TAG%"=="" (
  curl -fsSL --max-time 25 --retry 2 "https://github.com/%PBS_REPO%/releases.atom" -o "%TEMP%\gargantua_atom.xml" 2>nul
  if exist "%TEMP%\gargantua_atom.xml" (
    for /f "delims=" %%l in ('findstr /c:"%TAG_PREFIX%" "%TEMP%\gargantua_atom.xml"') do (
      if not defined TAG (
        set "LINE=%%l"
        set "TAG=!LINE:*%TAG_PREFIX%=!"
        set "TAG=!TAG:~0,8!"
      )
    )
    del "%TEMP%\gargantua_atom.xml" 2>nul
  )
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
      for /f "delims=" %%c in ('curl -s -o NUL -w "%%{http_code}" -I --connect-timeout 10 --max-time 30 "!PROBE_URL!" 2^>nul') do set "CODE=%%c"
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

REM ---- 3) 下载：自动测速选镜像（顺序探测前 2MB，取 speed_download 最大）----
set "ASSET=cpython-%VERSION%+%TAG%-%TRIPLE%-install_only.tar.gz"
if not exist "%PYTHON_DIR%\python\python.exe" (
  set "DIRECT=https://github.com/%PBS_REPO%/releases/download/%TAG%/%ASSET%"
  if not "%GARGANTUA_PY_BASE_URL%"=="" (
    set "MIRROR=%GARGANTUA_PY_BASE_URL%"
    echo [setup] 使用指定镜像：!MIRROR!
  ) else (
    set "SJTU=https://mirror.sjtu.edu.cn/github-release/%PBS_REPO%/releases/download/%TAG%/%ASSET%"
    set "BEST_URL="
    set "BEST_SPEED=0"
    for %%u in ("https://gh-proxy.com/!DIRECT!" "!DIRECT!" "!SJTU!") do (
      set "SPEED="
      for /f "delims=" %%s in ('curl -sL --max-time 8 -r 0-2097151 -o NUL -w "%%{speed_download}" "%%~u" 2^>nul') do set "SPEED=%%s"
      if not defined SPEED set "SPEED=0"
      echo [setup]   测速 !SPEED! B/s ^<- %%~u
      if !SPEED! gtr !BEST_SPEED! (
        set "BEST_SPEED=!SPEED!"
        set "BEST_URL=%%~u"
      )
    )
    if defined BEST_URL (
      set "MIRROR=!BEST_URL!"
    ) else (
      set "MIRROR=https://gh-proxy.com/!DIRECT!"
    )
    echo [setup] 选中镜像：!MIRROR!
  )

  if exist "%TEMP%\gargantua_py.tar.gz" del "%TEMP%\gargantua_py.tar.gz" 2>nul
  set "DL_OK=0"
  curl -fSL --retry 3 --retry-delay 2 -C - --max-time 900 "!MIRROR!" -o "%TEMP%\gargantua_py.tar.gz"
  if errorlevel 1 (
    echo [setup] 主镜像失败，按默认顺序重试……
  ) else (
    tar -tzf "%TEMP%\gargantua_py.tar.gz" >nul 2>&1
    if not errorlevel 1 set "DL_OK=1"
  )
  if "!DL_OK!"=="0" (
    for %%m in ("https://gh-proxy.com/!DIRECT!" "!DIRECT!" "!SJTU!") do (
      if "!DL_OK!"=="0" (
        if not "%%~m"=="!MIRROR!" (
          echo [setup] 尝试 %%~m
          del "%TEMP%\gargantua_py.tar.gz" 2>nul
          curl -fSL --retry 2 -C - --max-time 900 "%%~m" -o "%TEMP%\gargantua_py.tar.gz"
          if not errorlevel 1 (
            tar -tzf "%TEMP%\gargantua_py.tar.gz" >nul 2>&1
            if not errorlevel 1 (
              set "DL_OK=1"
              set "MIRROR=%%~m"
            )
          )
        )
      )
    )
  )
  if "!DL_OK!"=="0" (
    echo [setup] X 所有镜像下载失败。请检查网络/代理，或设 GARGANTUA_PY_BASE_URL 指向可达镜像。
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
  echo [setup] 使用指定 pip 源：%PIP_INDEX_URL%
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
