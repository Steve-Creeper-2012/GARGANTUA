@echo off
REM ===========================================================================
REM GARGANTUA v1 — 启动.bat（项目根目录，中文名）：Windows 一键启动
REM ---------------------------------------------------------------------------
REM 薄包装：中文名启动入口，直接调用 start\terminal\start.bat
REM （含环境安装、拉起 server\server.py）。用法同 start.bat：
REM   start.bat                正常启动（独立窗口）
REM   start.bat --no-window    仅 HTTP server
REM ===========================================================================
setlocal
call "%~dp0start\terminal\start.bat" %*
endlocal
