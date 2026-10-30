@echo off
REM ===========================================================================
REM GARGANTUA v1 — start/terminal/start.bat：Windows 一键启动
REM ---------------------------------------------------------------------------
REM 1. 调 setup.bat 确保隔离 Python 3.12.x + venv + 依赖（幂等；
REM    镜像自动测速选取）；
REM 2. 把 server\server.py **交给一个独立进程**（start /MIN，自己的控制台），
REM    然后本窗口立刻结束——终端不再被 server 占着，关掉启动窗口也不会
REM    带走 server。环境预检、窗口拉起、服务按需调度仍全部由 server 负责。
REM
REM 注：这里不用 pythonw.exe。pythonw 没有控制台，sys.stdout 是 None，
REM     server 里任何一句 print() 都会抛 AttributeError；独立的最小化
REM     控制台窗口既保住 stdout，也方便排查。
REM
REM 用法（命令提示符 / PowerShell）：
REM   start.bat                      交接给后台 server（独立窗口），随后退出
REM   start.bat --no-window          仅 HTTP server，不起窗口
REM   停止：任务管理器结束 server.py，或 curl -X POST http://127.0.0.1:8712/api/system/kill
REM ===========================================================================
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"

call "%SCRIPT_DIR%setup.bat"
if errorlevel 1 exit /b 1

set "PY=%ROOT%\environment\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [launcher] x 隔离 Python 不可用：%PY%
  exit /b 1
)

echo [launcher] 交接给 server ...
start "GARGANTUA server" /MIN "%PY%" "%ROOT%\server\server.py" %*

echo [launcher] OK  server 已在后台启动（最小化窗口），本窗口即将关闭。
endlocal
exit /b 0
