@echo off
REM ===========================================================================
REM GARGANTUA v1 — start/terminal/setup.bat：隔离环境引导（Windows）
REM ---------------------------------------------------------------------------
REM 与 launcher/setup-environment.sh 完全同构：
REM   * 只用 environment\python\ 里下载的 standalone CPython，绝不探测系统
REM     python / py -3.12，不用 winget，不装任何系统级包；
REM   * 空 environment\ 也能跑：用 Windows 自带 curl.exe / tar.exe 做一次性
REM     引导下载（只读取系统二进制，不安装、不写系统目录）；
REM   * 所有下载、缓存、venv、pip、工具链 HOME 全部落在项目文件夹内；
REM   * 镜像自动测速选最快；压缩包放进 environment\_cache\downloads\ 可离线装。
REM 其余逻辑全在跨平台的 start\setup_env.py（由项目内 Python 执行）。
REM ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
set "PY=%ROOT%\environment\python\python\python.exe"
set "CACHE=%ROOT%\environment\_cache\downloads"
set "CERTS=%ROOT%\environment\certs"
set "SETUP_PY=%ROOT%\start\setup_env.py"
if not exist "%CACHE%" mkdir "%CACHE%"
if not exist "%CERTS%" mkdir "%CERTS%"
if not exist "%ROOT%\environment\python" mkdir "%ROOT%\environment\python"

REM 启动只装必需项；可选二进制走 --stage tools 或窗口内按需安装
set "STAGE=runtime"
set "CHECK=0"
:parse
if "%~1"=="" goto :parsed
if "%~1"=="--check" set "CHECK=1" & shift & goto :parse
if "%~1"=="--stage" set "STAGE=%~2" & shift & shift & goto :parse
shift & goto :parse
:parsed

REM ---- 1) 环境检查：不合规时由 start.bat 调用本脚本重建 ----
if "%CHECK%"=="1" (
  if not exist "%PY%" exit /b 1
  "%PY%" "%SETUP_PY%" --check
  exit /b !errorlevel!
)

REM ---- 2) 引导下载 standalone Python（空 environment\ 也能跑）----
if exist "%PY%" goto :have_python
set "TAG=20261026"
if not "%GARGANTUA_PY_RELEASE%"=="" set "TAG=%GARGANTUA_PY_RELEASE%"
set "ASSET=cpython-3.12.16+%TAG%-x86_64-pc-windows-msvc-install_only.tar.gz"
if not "%GARGANTUA_PY_ASSET%"=="" set "ASSET=%GARGANTUA_PY_ASSET%"
echo [setup] 空环境：准备 standalone Python（全部下载到项目内）
echo [setup] 资产：!ASSET!

REM release 清单仅用于解析确切资产名；拿不到就用内置名
set "MANIFEST=%CACHE%\pbs_release.json"
if not exist "%MANIFEST%" (
  curl.exe -fsS -m 90 "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest" -o "%MANIFEST%" 2>nul
  if errorlevel 1 del "%MANIFEST%" 2>nul
)
if exist "%MANIFEST%" if "%GARGANTUA_PY_ASSET%"=="" (
  for /f "usebackq delims=" %%l in (`findstr /r /c:"cpython-3\.12\.[0-9][0-9]*+[0-9]*-x86_64-pc-windows-msvc-install_only\.tar\.gz" "%MANIFEST%"`) do (
    echo [setup] release 清单可用（资产名以 setup_env.py 解析为准）
  )
)

REM ---- 镜像测速：前 1MB 的下载速率，取最快 ----
set "M1=https://github.com/astral-sh/python-build-standalone/releases/download/%TAG%"
set "M2=https://gh-proxy.com/!M1!"
set "M3=https://mirror.sjtu.edu.cn/github-release/astral-sh/python-build-standalone/releases/download/%TAG%"
set "BEST="
set /a "BEST_SPEED=0"
for %%m in ("!M1!" "!M2!" "!M3!") do (
  set "SPEED=0"
  for /f "delims=." %%s in ('curl.exe -sL -m 8 -r 0-1048575 -o NUL -w "%%{speed_download}" "%%~m/!ASSET!" 2^>nul') do set "SPEED=%%s"
  if not defined SPEED set "SPEED=0"
  echo [setup]   测速 !SPEED! B/s  ^<- %%~m
  if !SPEED! GTR !BEST_SPEED! (
    set /a "BEST_SPEED=!SPEED!"
    set "BEST=%%~m"
  )
)

if exist "%CACHE%\!ASSET!" (
  echo [setup] 复用项目内缓存（离线）：!ASSET!
) else (
  if not defined BEST set "BEST=!M1!"
  echo [setup] 选中镜像：!BEST!
  curl.exe -fSL --retry 3 -m 900 "!BEST!/!ASSET!" -o "%CACHE%\!ASSET!.part"
  if errorlevel 1 (
    echo [setup] X standalone Python 下载失败
    exit /b 1
  )
  move /y "%CACHE%\!ASSET!.part" "%CACHE%\!ASSET!" >nul
)
tar.exe -xzf "%CACHE%\!ASSET!" -C "%ROOT%\environment\python"
if errorlevel 1 (
  echo [setup] X 解包失败
  exit /b 1
)
:have_python

if not exist "%PY%" (
  echo [setup] X 项目内 standalone Python 不可用：%PY%
  exit /b 1
)
echo [setup] 项目内 Python OK
"%PY%" "%SETUP_PY%" --stage %STAGE%
if errorlevel 1 (
  echo [setup] X 环境安装失败（详见上方）
  exit /b 1
)
echo [setup] OK 隔离环境就绪
endlocal
exit /b 0
