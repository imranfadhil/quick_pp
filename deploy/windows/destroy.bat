@echo off
setlocal enabledelayedexpansion

REM ===================================================================
REM  destroy.bat - Tear down the DigitalOcean droplet (+ SSH key)
REM  WARNING: permanently deletes the droplet and all data on it, including
REM           any OTHER app you host on that same droplet.
REM ===================================================================

cd /d "%~dp0"

if not exist "config.bat" (
  echo [destroy] config.bat not found. Run deploy.bat first.
  exit /b 1
)
call config.bat

echo.
echo ===============================================================
echo    DESTROY - quick_pp UI (DigitalOcean)
echo ===============================================================
echo.

set /p CONFIRM="Type DESTROY to confirm: "
if not "%CONFIRM%"=="DESTROY" (
    echo Aborted.
    exit /b 0
)

REM -- Shared-droplet guard --------------------------------------------
REM Compare the host's total container count with the count belonging to
REM THIS compose project (docker) and require the droplet name to be typed
REM as a second confirmation when foreign containers are found.
set PUBLIC_IP=
for /f %%i in ('doctl compute droplet get %DROPLET_NAME% --format PublicIPv4 --no-header 2^>nul') do set PUBLIC_IP=%%i
if "%PUBLIC_IP%"=="" goto :no_guard
if not exist "%KEY_FILE%" goto :no_guard

set "SSH_OPTS=-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes -i "%KEY_FILE%""
set "PROJECT_SLUG=docker"
set TOTAL_COUNT=0
set OUR_COUNT=0
for /f %%i in ('ssh %SSH_OPTS% root@%PUBLIC_IP% "docker ps -qa | wc -l" 2^>nul') do set TOTAL_COUNT=%%i
for /f %%i in ('ssh %SSH_OPTS% root@%PUBLIC_IP% "docker ps -qa --filter label=com.docker.compose.project=%PROJECT_SLUG% | wc -l" 2^>nul') do set OUR_COUNT=%%i
if %TOTAL_COUNT% leq %OUR_COUNT% goto :no_guard

echo.
echo [WARN] %DROPLET_NAME% is also running containers from OTHER projects:
ssh %SSH_OPTS% root@%PUBLIC_IP% "docker ps -a --format {{.Names}}" 2>nul
echo [WARN] Deleting %DROPLET_NAME% deletes them and all their data too.
echo.
set /p CONFIRM2="Type the droplet name (%DROPLET_NAME%) to delete it anyway: "
if /i not "%CONFIRM2%"=="%DROPLET_NAME%" (
    echo Aborted - nothing was deleted.
    exit /b 0
)

:no_guard

REM -- Delete droplet -------------------------------------------------
echo [1/2] Deleting droplet %DROPLET_NAME%...
doctl compute droplet get %DROPLET_NAME% >nul 2>&1
if %ERRORLEVEL% equ 0 (
    doctl compute droplet delete %DROPLET_NAME% --force >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo [OK] Droplet deleted.
    ) else (
        echo [WARN] Failed to delete droplet - check the DO console.
    )
) else (
    echo [SKIP] No droplet named %DROPLET_NAME% found.
)

REM -- Delete SSH key from DO (keeps the local key file) --------------
echo [2/2] Removing SSH key from DO...
set KEY_ID=
for /f "tokens=1,2" %%i in ('doctl compute ssh-key list --format ID^,Name --no-header') do (
    if /i "%%j"=="%KEY_NAME%" set KEY_ID=%%i
)
if not "%KEY_ID%"=="" (
    doctl compute ssh-key delete %KEY_ID% --force >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo [OK] SSH key removed from DO ^(%KEY_NAME%, ID: %KEY_ID%^). Local key file kept at %KEY_FILE%.
    ) else (
        echo [WARN] Failed to remove SSH key from DO ^(ID: %KEY_ID%^).
        echo        Remove manually: doctl compute ssh-key delete %KEY_ID%
    )
) else (
    echo [SKIP] No SSH key named %KEY_NAME% on DO.
)

REM -- Clean cached state ---------------------------------------------
del "%TEMP%\%PROJECT_NAME%-do-ip.txt" 2>nul

echo.
echo ===============================================================
echo    DESTROY COMPLETE
echo ===============================================================

endlocal

