@echo off
REM Boot both services for local development on Windows.
REM Opens two new console windows so you can see each service's logs separately.
REM Close either window (or hit Ctrl+C inside it) to stop that service.

setlocal
set PROJECT_ROOT=%~dp0
set VENV_PY=%PROJECT_ROOT%venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    echo ERROR: venv python not found at %VENV_PY%
    echo Run: python -m venv venv ^&^& venv\Scripts\pip install -r api\requirements.txt -r web\requirements.txt
    exit /b 1
)

REM Silence MediaPipe's native clearcut telemetry retries.
set GLOG_minloglevel=3

echo Starting FastAPI on http://127.0.0.1:8001 ...
start "Fashion Try-On API" /D "%PROJECT_ROOT%" "%VENV_PY%" -m uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload

echo Starting Django on  http://127.0.0.1:8000 ...
start "Fashion Try-On Web" /D "%PROJECT_ROOT%web" "%VENV_PY%" manage.py runserver 127.0.0.1:8000

echo.
echo Both services launched in new console windows.
echo Open http://127.0.0.1:8000/ in your browser.
echo Close either window or press Ctrl+C inside it to stop that service.

endlocal
