@echo off
setlocal enabledelayedexpansion

REM ===================================================================
REM  deploy.bat - Provision quick_pp UI on a DigitalOcean Droplet
REM  Creates the droplet with doctl (no manual IP needed), then pushes
REM  the code and starts the Docker stack.
REM
REM  Prereqs: doctl (authenticated), ssh + scp + ssh-keygen (OpenSSH), tar
REM ===================================================================

cd /d "%~dp0"
set DEPLOY_DIR=%CD%

if not exist "config.bat" (
  echo [deploy] config.bat not found - seeding from .env via config.example.bat
  call "%DEPLOY_DIR%\seed-config.bat"
  if not exist "config.bat" (
    echo [deploy] Could not create config.bat. Edit deploy\config.bat, then re-run.
    exit /b 1
  )
  echo [deploy] Review deploy\config.bat ^(check SIZE, REGION, DOMAIN^), then re-run deploy.bat.
  exit /b 1
)
call config.bat

echo.
echo ===============================================================
echo    QUICK_PP UI - DIGITALOCEAN DEPLOY
echo ===============================================================
echo.

REM -- Prerequisites --------------------------------------------------
where doctl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] doctl not found. Install: https://docs.digitalocean.com/reference/doctl/how-to/install/
    exit /b 1
)
doctl account get >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] doctl not authenticated. Run: doctl auth init
    exit /b 1
)
echo [OK] doctl authenticated.

REM -- Step 1: Ensure a local SSH key + register it with DO -----------
echo.
echo [1/5] Ensuring SSH key...
if not exist "%KEY_FILE%" (
    mkdir "%USERPROFILE%\.ssh" 2>nul
    ssh-keygen -t ed25519 -N "" -f "%KEY_FILE%" -C "%KEY_NAME%"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] ssh-keygen failed.
        exit /b 1
    )
    echo [OK] Key generated: %KEY_FILE%
) else (
    echo [OK] Local key exists: %KEY_FILE%
)

REM Register (import) the public key with DO if not already there.
set KEY_FP=
for /f "tokens=1,2" %%a in ('doctl compute ssh-key list --format Name^,FingerPrint --no-header') do (
    if /i "%%a"=="%KEY_NAME%" set KEY_FP=%%b
)
if not "%KEY_FP%"=="" (
    echo [OK] SSH key already registered with DO ^(by name^).
) else (
    REM Not found by name - compute local MD5 fingerprint and search DO.
    for /f "tokens=2" %%f in ('ssh-keygen -lf "%PUB_FILE%" -E md5') do set LOCAL_FP=%%f
    set LOCAL_FP=!LOCAL_FP:MD5:=!
    for /f "tokens=1,2" %%a in ('doctl compute ssh-key list --format Name^,FingerPrint --no-header') do (
        if /i "%%b"=="!LOCAL_FP!" set KEY_FP=%%b
    )
    if not "!KEY_FP!"=="" (
        echo [OK] SSH key already registered with DO ^(matched by fingerprint^).
    ) else (
        doctl compute ssh-key import %KEY_NAME% --public-key-file "%PUB_FILE%" >nul
        if !ERRORLEVEL! neq 0 (
            echo [ERROR] Failed to import SSH key to DO.
            echo        The key may already exist under a different name.
            echo        Check: doctl compute ssh-key list
            exit /b 1
        )
        echo [OK] SSH key registered with DO.
        for /f "tokens=1,2" %%a in ('doctl compute ssh-key list --format Name^,FingerPrint --no-header') do (
            if /i "%%a"=="%KEY_NAME%" set KEY_FP=%%b
        )
    )
)

if "%KEY_FP%"=="" (
    echo [ERROR] Could not resolve SSH key fingerprint.
    exit /b 1
)

REM -- Step 2: Create the droplet ------------------------------------
echo.
echo [2/5] Creating droplet...

doctl compute droplet get %DROPLET_NAME% >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Droplet already exists: %DROPLET_NAME%
    goto :get_ip
)

set USERDATA_TMP=%TEMP%\do-userdata-unix.sh
powershell -NoProfile -Command "(Get-Content '%DEPLOY_DIR%\userdata.sh' -Raw) -replace [char]13, '' | Set-Content -NoNewline '%USERDATA_TMP%' -Encoding ascii"

doctl compute droplet create %DROPLET_NAME% ^
    --region %REGION% ^
    --size %SIZE% ^
    --image %IMAGE% ^
    --ssh-keys %KEY_FP% ^
    --user-data-file "%USERDATA_TMP%" ^
    --wait
set CREATE_RC=%ERRORLEVEL%
del "%USERDATA_TMP%" 2>nul
if not %CREATE_RC% equ 0 (
    echo [ERROR] Droplet creation failed.
    exit /b 1
)
echo [OK] Droplet created: %DROPLET_NAME%

:get_ip
echo.
echo [3/5] Resolving public IP...
for /f %%i in ('doctl compute droplet get %DROPLET_NAME% --format PublicIPv4 --no-header') do set PUBLIC_IP=%%i
if "%PUBLIC_IP%"=="" (
    echo [ERROR] Could not resolve droplet public IP.
    exit /b 1
)
echo [OK] Public IP: %PUBLIC_IP%
echo %PUBLIC_IP% > "%TEMP%\%PROJECT_NAME%-do-ip.txt"

REM -- Step 4: Wait for Docker/boot to settle ------------------------
echo.
echo [4/5] Waiting for the droplet to finish booting...
timeout /t 45 /nobreak >nul

REM -- Step 5: Push code and start the app ---------------------------
echo.
echo [5/5] Deploying code...
call "%DEPLOY_DIR%\sync.bat" --from-deploy
if %ERRORLEVEL% neq 0 (
    echo [WARN] Initial sync failed - droplet may still be booting.
    echo        Wait a minute and run:  sync.bat --restart
)

echo.
echo ===============================================================
echo    DIGITALOCEAN DEPLOY COMPLETE
echo ===============================================================
echo    Droplet:    %DROPLET_NAME%  (%SIZE%, %REGION%)
echo    Public IP:  %PUBLIC_IP%
echo    Key File:   %KEY_FILE%
echo.
if not "%DOMAIN%"=="" (
    echo    App:        https://%DOMAIN%/
    echo.
)
echo    SSH:        ssh -i "%KEY_FILE%" root@%PUBLIC_IP%
echo ===============================================================

endlocal
