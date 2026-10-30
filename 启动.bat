@echo off
REM ===========================================================================
REM GARGANTUA v1 — 启动.bat（项目根目录，中文名）：Windows 一键启动
REM ---------------------------------------------------------------------------
REM 薄包装：中文名启动入口，直接调用 start\terminal\start.bat
REM （环境安装 → 把 server 交给独立进程 → 本窗口结束）。用法同 start.bat：
REM   start.bat                交接给后台 server（独立窗口），随后退出
REM   start.bat --no-window    仅 HTTP server
REM ===========================================================================
setlocal
call "%~dp0start\terminal\start.bat" %*
endlocal
