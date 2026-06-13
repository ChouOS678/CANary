@echo off
chcp 65001>nul
cd /d %~dp0

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m venv .venv
  call .venv\Scripts\activate.bat
) else (
  python -m venv .venv
  call .venv\Scripts\activate.bat
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py

echo.
echo 输出已生成到 output 目录。
pause
