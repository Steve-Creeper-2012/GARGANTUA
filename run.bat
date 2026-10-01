@echo off
REM ===========================================================================
REM GARGANTUA v1 — Windows 一键启动
REM 用法：
REM   run.bat                  启动网页工作台 http://127.0.0.1:8712
REM   run.bat --port 9000      指定端口
REM   run.bat --no-open        不自动打开浏览器
REM   run.bat --test           跑全量测试
REM   run.bat --init NAME      创建模型 models\NAME
REM   run.bat --train NAME     用 data\ 训练 models\NAME
REM   run.bat --infer NAME     推理（提示词先 set PROMPT=...）
REM ===========================================================================
setlocal
cd /d "%~dp0"

REM 优先项目 venv
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  REM 其次找系统 python
  where python >nul 2>nul && (set "PY=python") || (
    echo [run.bat] 找不到 Python，请先安装 Python 3.10+ 并加入 PATH
    exit /b 1
  )
)

"%PY%" run.py %*
endlocal
