@echo off
REM Stop whatever is listening on ports 8000 (Django) and 8001 (FastAPI).
REM Useful as a backup when closing the run.bat windows didn't fully clean up.

setlocal EnableDelayedExpansion

set API_PORT=8001
set WEB_PORT=8000

call :kill_port %API_PORT%
call :kill_port %WEB_PORT%

endlocal
exit /b 0

:kill_port
set PORT=%~1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":%PORT% "') do (
    echo [stop] killing PID %%a (port %PORT%)
    taskkill /F /T /PID %%a >nul 2>&1
)
exit /b 0
