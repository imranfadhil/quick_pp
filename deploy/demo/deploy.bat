@echo off
setlocal enabledelayedexpansion

REM ===================================================================
REM  deploy.bat - Deploy quick_pp Interactive Live Demo to a Shared or
REM               Small DigitalOcean Droplet
REM
REM  If SHARED_DROPLET_IP is set or DROPLET_NAME already exists on DO,
REM  it reuses that droplet without creating a new one and deploys into
REM  /opt/quick_pp_demo on port 7860.
REM ===================================================================

cd /d "%~dp0"
set DEPLOY_DIR=%CD%

if not exist "config.bat" (
  echo [deploy-demo] config.bat not found - generating from config.example.bat
  call "%DEPLOY_DIR%\seed-config.bat"
  if not exist "config.bat" (
    echo [deploy-demo] Could not create config.bat. Edit deploy\demo\config.bat, then re-run.
    exit /b 1
  )
  echo [deploy-demo] Review deploy\demo\config.bat ^(set DROPLET_NAME or SHARED_DROPLET_IP^), then re-run deploy.bat.
  exit /b 1
)
call config.bat

echo.
echo ===============================================================
echo    QUICK_PP INTERACTIVE LIVE DEMO - DROPLET DEPLOY
echo ===============================================================
echo.

REM -- Check if SHARED_DROPLET_IP is explicitly provided --------------
if not "%SHARED_DROPLET_IP%"=="" (
    set PUBLIC_IP=%SHARED_DROPLET_IP%
    echo [OK] Using configured SHARED_DROPLET_IP: !PUBLIC_IP!
    echo !PUBLIC_IP! > "%TEMP%\%PROJECT_NAME%-do-ip.txt"
    goto :deploy_code
)

REM -- Prerequisites for doctl ----------------------------------------
where doctl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] doctl not found. Install doctl or set SHARED_DROPLET_IP in config.bat.
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
echo [1/4] Ensuring SSH key...
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

set KEY_FP=
for /f "tokens=1,2" %%a in ('doctl compute ssh-key list --format Name^,FingerPrint --no-header') do (
    if /i "%%a"=="%KEY_NAME%" set KEY_FP=%%b
)
if not "%KEY_FP%"=="" (
    echo [OK] SSH key already registered with DO ^(by name^).
) else (
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
            exit /b 1
        )
        echo [OK] SSH key registered with DO.
        for /f "tokens=1,2" %%a in ('doctl compute ssh-key list --format Name^,FingerPrint --no-header') do (
            if /i "%%a"=="%KEY_NAME%" set KEY_FP=%%b
        )
    )
)

REM -- Step 2: Check existing shared droplet or create small droplet ---
echo.
echo [2/4] Checking droplet "%DROPLET_NAME%"...

doctl compute droplet get %DROPLET_NAME% >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Reusing existing droplet: %DROPLET_NAME%
    goto :get_ip
)

set USERDATA_TMP=%TEMP%\do-demo-userdata-unix.sh
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
echo [OK] Small droplet created: %DROPLET_NAME% (%SIZE%)

echo.
echo [INFO] Waiting 40s for new droplet cloud-init & Docker daemon...
timeout /t 40 /nobreak >nul

:get_ip
echo.
echo [3/4] Resolving public IP...
for /f %%i in ('doctl compute droplet get %DROPLET_NAME% --format PublicIPv4 --no-header') do set PUBLIC_IP=%%i
if "%PUBLIC_IP%"=="" (
    echo [ERROR] Could not resolve droplet public IP.
    exit /b 1
)
echo [OK] Public IP: %PUBLIC_IP%
echo %PUBLIC_IP% > "%TEMP%\%PROJECT_NAME%-do-ip.txt"

:deploy_code
echo.
echo [4/4] Syncing and starting isolated demo container...
call "%DEPLOY_DIR%\sync.bat" --from-deploy
if %ERRORLEVEL% neq 0 (
    echo [WARN] Initial sync failed - if droplet is newly created, wait 30s and run:
    echo        deploy\demo\sync.bat --restart
    exit /b 1
)

echo.
echo ===============================================================
echo    QUICK_PP LIVE DEMO DEPLOY COMPLETE
echo ===============================================================
echo    Droplet:     %DROPLET_NAME%
echo    Public IP:   %PUBLIC_IP%
echo    Remote Dir:  %REMOTE_DIR%
echo    Host Port:   %DEMO_BIND%:%DEMO_PORT% -^> container :7860
echo    SSH:         ssh -i "%KEY_FILE%" root@%PUBLIC_IP%
echo ===============================================================

endlocal

