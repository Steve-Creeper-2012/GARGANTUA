@echo off
REM ===========================================================================
REM GARGANTUA v1 — start/terminal/start.bat：Windows 一键启动
REM ---------------------------------------------------------------------------
REM 1. 调 setup.bat 确保隔离 Python 3.12.x + venv + 依赖（幂等）；
REM 2. 用隔离 venv 的解释器启动 server\server.py。
REM    环境检查、window.py 完整性校验（sha256 + 弹窗）、窗口拉起、
REM    服务按需调度全部由 server 负责。
REM
REM 用法（命令提示符 / PowerShell）：
REM   start.bat                      正常启动（独立窗口 + 完整性弹窗）
REM   start.bat --no-window          仅 HTTP server，不起窗口
REM   start.bat --no-verify          跳过窗口完整性弹窗
REM ===========================================================================
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"

call "%SCRIPT_DIR%setup.bat"
if errorlevel 1 exit /b 1

"%ROOT%\start\.venv\Scripts\python.exe" "%ROOT%\server\server.py" %*
endlocal
