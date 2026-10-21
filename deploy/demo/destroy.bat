@echo off
setlocal enabledelayedexpansion

REM ===================================================================
REM  destroy.bat - Tear down the quick_pp Live Demo
REM
REM  Safe for shared droplets! Lets you choose between:
REM    1. APP ONLY (Recommended for shared droplets): Stops and removes
REM       only the quick_pp_demo containers and /opt/quick_pp_demo
REM       directory, leaving the droplet and other apps untouched.
REM    2. ENTIRE DROPLET: Deletes the DigitalOcean droplet itself.
REM ===================================================================

cd /d "%~dp0"

if not exist "config.bat" (
  echo [destroy-demo] config.bat not found.
  exit /b 1
)
call config.bat

echo.
echo ===============================================================
echo    TEARDOWN - quick_pp Interactive Live Demo
echo ===============================================================
echo.
echo Choose teardown mode:
echo   [1] Remove DEMO APP ONLY ^(safe for shared droplets - keeps droplet^)
echo   [2] Destroy ENTIRE DROPLET ^(%DROPLET_NAME%^)
echo.
set /p MODE="Select 1 or 2 (or N to cancel): "

if "%MODE%"=="1" goto :remove_app_only
if "%MODE%"=="2" goto :destroy_droplet
echo Aborted.
exit /b 0

:remove_app_only
set PUBLIC_IP=%SHARED_DROPLET_IP%
if "%PUBLIC_IP%"=="" (
    if exist "%TEMP%\%PROJECT_NAME%-do-ip.txt" set /p PUBLIC_IP=<"%TEMP%\%PROJECT_NAME%-do-ip.txt"
)
if "%PUBLIC_IP%"=="" (
    for /f %%i in ('doctl compute droplet get %DROPLET_NAME% --format PublicIPv4 --no-header 2^>nul') do set PUBLIC_IP=%%i
)
if "%PUBLIC_IP%"=="" (
    echo [ERROR] Could not resolve droplet IP.
    exit /b 1
)
set "SSH_OPTS=-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i "%KEY_FILE%""
echo.
echo [1/2] Stopping and removing %COMPOSE_PROJECT% containers & images on %PUBLIC_IP%...
ssh %SSH_OPTS% root@%PUBLIC_IP% "docker compose -p %COMPOSE_PROJECT% -f %REMOTE_DIR%/deploy/demo/docker-compose.yaml down --rmi local -v 2>/dev/null || true"
echo [2/2] Removing %REMOTE_DIR%...
ssh %SSH_OPTS% root@%PUBLIC_IP% "rm -rf %REMOTE_DIR%"
echo.
echo [OK] Demo app removed from %PUBLIC_IP%. Shared droplet and other apps are untouched.
exit /b 0

:destroy_droplet
set /p CONFIRM="Type DESTROY to permanently delete droplet %DROPLET_NAME%: "
if not "%CONFIRM%"=="DESTROY" (
    echo Aborted.
    exit /b 0
)

set PUBLIC_IP=
for /f %%i in ('doctl compute droplet get %DROPLET_NAME% --format PublicIPv4 --no-header 2^>nul') do set PUBLIC_IP=%%i
if "%PUBLIC_IP%"=="" goto :no_guard
if not exist "%KEY_FILE%" goto :no_guard

set "SSH_OPTS=-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes -i "%KEY_FILE%""
set TOTAL_COUNT=0
set OUR_COUNT=0
for /f %%i in ('ssh %SSH_OPTS% root@%PUBLIC_IP% "docker ps -qa | wc -l" 2^>nul') do set TOTAL_COUNT=%%i
for /f %%i in ('ssh %SSH_OPTS% root@%PUBLIC_IP% "docker ps -qa --filter label=com.docker.compose.project=%COMPOSE_PROJECT% | wc -l" 2^>nul') do set OUR_COUNT=%%i
if %TOTAL_COUNT% leq %OUR_COUNT% goto :no_guard

echo.
echo [WARN] %DROPLET_NAME% is also running containers from OTHER projects:
ssh %SSH_OPTS% root@%PUBLIC_IP% "docker ps -a --format {{.Names}}" 2>nul
echo [WARN] Deleting %DROPLET_NAME% deletes them and all their data too!
echo.
set /p CONFIRM2="Type the droplet name (%DROPLET_NAME%) to delete it anyway: "
if /i not "%CONFIRM2%"=="%DROPLET_NAME%" (
    echo Aborted - nothing was deleted.
    exit /b 0
)

:no_guard
echo Deleting droplet %DROPLET_NAME%...
doctl compute droplet delete %DROPLET_NAME% --force
del "%TEMP%\%PROJECT_NAME%-do-ip.txt" 2>nul
echo [OK] Droplet %DROPLET_NAME% deleted.
endlocal

