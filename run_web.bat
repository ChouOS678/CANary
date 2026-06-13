@echo off
setlocal
cd /d %~dp0

set "VENV_DIR=.venv_web313"

where py >nul 2>nul || (
  echo [ERROR] Python launcher 'py' not found.
  pause
  exit /b 1
)

py -3.13 -c "import sys" >nul 2>nul || (
  echo [ERROR] Python 3.13 is not available on this machine.
  pause
  exit /b 1
)

py -3.13 -m venv "%VENV_DIR%" || (
  echo [ERROR] Failed to create Python 3.13 virtual environment.
  pause
  exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat" || (
  echo [ERROR] Failed to activate virtual environment.
  pause
  exit /b 1
)

python -m pip install --upgrade pip || (
  echo [ERROR] Failed to upgrade pip.
  pause
  exit /b 1
)

python -m pip install -r requirements.txt || (
  echo [ERROR] Failed to install dependencies for Python 3.13.
  pause
  exit /b 1
)

set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
(
  echo [general]
  echo email = ""
) > "%USERPROFILE%\.streamlit\credentials.toml"

python -m streamlit run web_app.py --server.headless false --server.address localhost || (
  echo [ERROR] Failed to start Streamlit.
  pause
  exit /b 1
)

endlocal
