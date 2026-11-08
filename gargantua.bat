@echo off
REM GARGANTUA v1 — 终端接口启动器（Windows）
REM 只用项目内 venv 的 Python，绝不回退系统解释器；路径全部由脚本自身位置推导。
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%environment\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo 请先运行 启动.bat 完成环境初始化
  exit /b 1
)
"%PY%" "%ROOT%service\cli\gargantua.py" %*
endlocal
