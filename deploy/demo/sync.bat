@echo off
setlocal enabledelayedexpansion

REM ===================================================================
REM  sync.bat - Push quick_pp Live Demo to a Shared or Small Droplet
REM
REM  Packages the app + Volve/Costa raw LAS/CSV seed files (~2.5 MB),
REM  ensures 2 GB swap is active on the droplet, builds/starts the
REM  isolated quick_pp_demo stack on port 7860, and prints the free
REM  Cloudflare Quick Tunnel HTTPS URL.
REM
REM  Usage:
REM    sync.bat              push code + restart demo container
REM    sync.bat --restart    force Docker image rebuild + restart
REM    sync.bat --from-deploy called by deploy.bat
REM ===================================================================

cd /d "%~dp0"
set DEPLOY_DIR=%CD%

if not exist "config.bat" (
  echo [sync-demo] config.bat not found - generating from config.example.bat
  call "%DEPLOY_DIR%\seed-config.bat"
  echo [sync-demo] Review deploy\demo\config.bat, then re-run.
  exit /b 1
)
call config.bat

set BUILD=0
if "%1"=="--restart"     set BUILD=1
if "%1"=="--from-deploy" set BUILD=1

echo.
echo === SYNC - quick_pp Interactive Live Demo ===
echo.

REM -- Resolve public IP ----------------------------------------------
set PUBLIC_IP=%SHARED_DROPLET_IP%
if "%PUBLIC_IP%"=="" (
    if exist "%TEMP%\%PROJECT_NAME%-do-ip.txt" set /p PUBLIC_IP=<"%TEMP%\%PROJECT_NAME%-do-ip.txt"
)
if "%PUBLIC_IP%"=="" (
    for /f %%i in ('doctl compute droplet get %DROPLET_NAME% --format PublicIPv4 --no-header 2^>nul') do set PUBLIC_IP=%%i
)
if "%PUBLIC_IP%"=="" (
    echo [ERROR] No droplet IP found. Set SHARED_DROPLET_IP in config.bat or run deploy.bat first.
    exit /b 1
)
echo [INFO] Target Droplet IP: %PUBLIC_IP%  ^(Remote path: %REMOTE_DIR%^)

if not exist "%KEY_FILE%" (
    echo [ERROR] SSH key not found: %KEY_FILE%
    exit /b 1
)

set "SSH_OPTS=-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -i "%KEY_FILE%""
set SSH_RETRIES=5

REM -- Package code + minimal Volve/Costa seed files -------------------
echo [1/5] Packaging demo project and Volve/Costa seed files...
cd /d "%~dp0..\.."
if exist "%TEMP%\%PROJECT_NAME%.tar" del "%TEMP%\%PROJECT_NAME%.tar"
tar -cf "%TEMP%\%PROJECT_NAME%.tar" ^
    --exclude="node_modules" ^
    --exclude=".svelte-kit" ^
    --exclude="quick_pp/app/frontend/build" ^
    --exclude=".git" ^
    --exclude=".venv" ^
    --exclude="venv" ^
    --exclude="__pycache__" ^
    --exclude=".pytest_cache" ^
    --exclude=".mypy_cache" ^
    --exclude="dist" ^
    --exclude="build" ^
    --exclude="*.egg-info" ^
    --exclude="docs" ^
    --exclude="data" ^
    --exclude="uploads" ^
    --exclude="mlruns" ^
    --exclude="quick_pp/app/docker/data" ^
    --exclude=".env" ^
    --exclude="deploy/demo/config.bat" ^
    --exclude="*.log" ^
    --exclude="notebooks/*.ipynb" ^
    --exclude="notebooks/data/02_interim" ^
    --exclude="notebooks/data/03_primary" ^
    --exclude="notebooks/data/04_project" ^
    pyproject.toml README.md quick_pp ^
    deploy/demo/Dockerfile ^
    deploy/demo/docker-compose.yaml ^
    deploy/demo/nginx.conf ^
    deploy/demo/start_demo.sh ^
    deploy/demo/seed_demo.py ^
    notebooks/data/01_raw/VOLVE ^
    notebooks/data/01_raw/COSTA ^
    notebooks/data/01_raw/36_7-3.las ^
    notebooks/data/01_raw/formation_pressure_data.csv
if %ERRORLEVEL% neq 0 (
    echo [ERROR] tar packaging failed.
    exit /b 1
)
echo [OK] Packaged.

REM -- Ensure remote directory + 2 GB swap on small/shared droplet -----
echo [2/5] Ensuring remote directory and swap file on droplet...
call :ssh_run "mkdir -p %REMOTE_DIR% && (swapon --show | grep -q '/swapfile' || (fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile)) || true"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Could not reach droplet over SSH.
    exit /b 1
)
echo [OK] Remote directory and swap verified.

REM -- Upload archive --------------------------------------------------
echo [3/5] Uploading archive to %REMOTE_DIR%...
call :ssh_pipe "%TEMP%\%PROJECT_NAME%.tar" "cat > %REMOTE_DIR%/demo_app.tar"
set SCP_RC=%ERRORLEVEL%
del "%TEMP%\%PROJECT_NAME%.tar" 2>nul
if not %SCP_RC% equ 0 (
    echo [ERROR] Upload failed.
    exit /b 1
)
echo [OK] Uploaded.

REM -- Extract and write demo .env -------------------------------------
echo [4/5] Extracting and configuring %REMOTE_DIR%/.env...
call :ssh_run "cd %REMOTE_DIR% && tar -xf demo_app.tar && rm -f demo_app.tar"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Extraction failed.
    exit /b 1
)

set "PUBLIC_URL_VAL="
if not "%DOMAIN%"=="" set "PUBLIC_URL_VAL=https://%DOMAIN%"

call :ssh_run "printf 'QPP_DEMO_MODE=true\nQPP_DISABLE_TRAINING=true\nDEMO_PORT=%DEMO_PORT%\nDEMO_BIND=%DEMO_BIND%\nPUBLIC_URL=%PUBLIC_URL_VAL%\n' > %REMOTE_DIR%/.env"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to write remote .env.
    exit /b 1
)
echo [OK] Demo environment configured.

REM -- Build & Start isolated demo compose stack -----------------------
echo [5/5] Starting isolated demo stack ^(project: %COMPOSE_PROJECT%^)...
set "COMPOSE_CMD=docker compose -p %COMPOSE_PROJECT% --env-file %REMOTE_DIR%/.env -f %REMOTE_DIR%/deploy/demo/docker-compose.yaml"

REM Rebuild/update only `qpp-demo` (--no-deps) and keep `demo-quicktunnel` running (--no-recreate --no-deps)
REM so the *.trycloudflare.com HTTPS URL does not change across sync.bat / sync.bat --restart runs.
if "%1"=="--from-deploy" (
    set "REMOTE_CMD=!COMPOSE_CMD! up -d --build"
) else if %BUILD% equ 1 (
    set "REMOTE_CMD=!COMPOSE_CMD! up -d --build --no-deps qpp-demo && !COMPOSE_CMD! up -d --no-recreate --no-deps demo-quicktunnel"
) else (
    set "REMOTE_CMD=!COMPOSE_CMD! up -d --no-recreate --no-deps qpp-demo demo-quicktunnel"
)

call :ssh_run "!REMOTE_CMD!"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Demo container startup failed. Inspect logs via:
    echo         ssh -i "%KEY_FILE%" root@%PUBLIC_IP% "!COMPOSE_CMD! logs"
    exit /b 1
)

if %BUILD% equ 0 (
    call :ssh_run "!COMPOSE_CMD! restart qpp-demo" >nul 2>&1
)

echo.
echo [OK] Demo stack running:
call :ssh_run "!COMPOSE_CMD! ps"

REM -- Retrieve Cloudflare Quick Tunnel URL ----------------------------
set QUICK_URL=
echo.
echo [INFO] Resolving Cloudflare Quick Tunnel HTTPS URL...
for /l %%n in (1,1,15) do (
    if "!QUICK_URL!"=="" (
        ssh %SSH_OPTS% root@%PUBLIC_IP% "!COMPOSE_CMD! logs demo-quicktunnel | grep -o https://[a-z0-9-]*\.trycloudflare\.com | tail -1" <nul > "%TEMP%\%PROJECT_NAME%-quick-url.txt" 2>nul
        for /f "usebackq delims=" %%u in ("%TEMP%\%PROJECT_NAME%-quick-url.txt") do set QUICK_URL=%%u
        if "!QUICK_URL!"=="" ping -n 3 127.0.0.1 >nul 2>&1
    )
)
del "%TEMP%\%PROJECT_NAME%-quick-url.txt" 2>nul

echo.
if not "!QUICK_URL!"=="" (
    echo ===============================================================
    echo    LIVE DEMO URL ^(HTTPS^): !QUICK_URL!
    echo ===============================================================
) else (
    echo [WARN] Quick tunnel URL not ready yet. Check logs with:
    echo        ssh -i "%KEY_FILE%" root@%PUBLIC_IP% "!COMPOSE_CMD! logs demo-quicktunnel"
)

goto :sync_end

:ssh_run
set /a SSH_TRY=0
:ssh_run_loop
set /a SSH_TRY+=1
ssh %SSH_OPTS% root@%PUBLIC_IP% %* <nul
if %ERRORLEVEL% equ 0 exit /b 0
if %SSH_TRY% geq %SSH_RETRIES% exit /b 1
echo [WARN] SSH attempt %SSH_TRY%/%SSH_RETRIES% failed - retrying...
ping -n 5 127.0.0.1 >nul
goto :ssh_run_loop

:ssh_pipe
set /a SSH_TRY=0
:ssh_pipe_loop
set /a SSH_TRY+=1
type "%~1" | ssh %SSH_OPTS% root@%PUBLIC_IP% %2
if %ERRORLEVEL% equ 0 exit /b 0
if %SSH_TRY% geq %SSH_RETRIES% exit /b 1
echo [WARN] SSH attempt %SSH_TRY%/%SSH_RETRIES% failed - retrying...
ping -n 5 127.0.0.1 >nul
goto :ssh_pipe_loop

:sync_end
endlocal

