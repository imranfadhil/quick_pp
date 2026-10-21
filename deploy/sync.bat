@echo off
setlocal enabledelayedexpansion

REM ===================================================================
REM  sync.bat - Push quick_pp code to the DigitalOcean droplet over SSH
REM  Resolves the droplet IP automatically via doctl (no manual IP).
REM
REM  Usage:
REM    sync.bat              push code + restart containers
REM    sync.bat --restart    force docker rebuild + restart frontend
REM    sync.bat --from-deploy called by deploy.bat (first push)
REM ===================================================================

cd /d "%~dp0"
set DEPLOY_DIR=%CD%

if not exist "config.bat" (
  echo [sync] config.bat not found - seeding from .env via config.example.bat
  call "%DEPLOY_DIR%\seed-config.bat"
  echo [sync] Review deploy\config.bat, then re-run.
  exit /b 1
)
call config.bat

set BUILD=0
if "%1"=="--restart"     set BUILD=1
if "%1"=="--from-deploy" set BUILD=1

echo.
echo === SYNC - quick_pp UI (DigitalOcean) ===
echo.

REM -- Resolve public IP ----------------------------------------------
set PUBLIC_IP=
if exist "%TEMP%\%PROJECT_NAME%-do-ip.txt" set /p PUBLIC_IP=<"%TEMP%\%PROJECT_NAME%-do-ip.txt"
if "%PUBLIC_IP%"=="" (
    for /f %%i in ('doctl compute droplet get %DROPLET_NAME% --format PublicIPv4 --no-header 2^>nul') do set PUBLIC_IP=%%i
)
if "%PUBLIC_IP%"=="" (
    echo [ERROR] No droplet IP found. Run deploy.bat first.
    exit /b 1
)
echo [INFO] Target: %PUBLIC_IP%

if not exist "%KEY_FILE%" (
    echo [ERROR] SSH key not found: %KEY_FILE%
    exit /b 1
)

set "SSH_OPTS=-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -i "%KEY_FILE%""
set SSH_RETRIES=5

REM -- Package code (exclude heavy local folders, venv, notebooks, data) --
echo [1/5] Packaging project...
cd /d "%~dp0.."
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
    --exclude="notebooks" ^
    --exclude="docs/_build" ^
    --exclude="data" ^
    --exclude="uploads" ^
    --exclude="mlruns" ^
    --exclude="quick_pp/app/docker/data" ^
    --exclude=".env" ^
    --exclude="deploy" ^
    --exclude="*.log" ^
    *
if %ERRORLEVEL% neq 0 (
    echo [ERROR] tar packaging failed.
    exit /b 1
)
echo [OK] Packaged.

REM -- Copy to droplet ------------------------------------------------
echo [2/5] Ensuring remote directory exists...
call :ssh_run "mkdir -p %REMOTE_DIR%/uploads"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Could not create remote directory. Droplet may still be booting.
    exit /b 1
)
echo [OK] Remote directory ready.

echo [3/5] Copying to droplet...
call :ssh_pipe "%TEMP%\%PROJECT_NAME%.tar" "cat > %REMOTE_DIR%/app.tar"
set SCP_RC=%ERRORLEVEL%
del "%TEMP%\%PROJECT_NAME%.tar" 2>nul
if not %SCP_RC% equ 0 (
    echo [ERROR] Upload failed. Droplet may still be booting, or SSH not ready yet.
    exit /b 1
)
echo [OK] Uploaded.

REM -- Extract + configure .env ---------------------------------------
echo [4/5] Extracting and configuring environment...
call :ssh_run "cd %REMOTE_DIR% && tar -xf app.tar && rm -f app.tar"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Extraction failed.
    exit /b 1
)

call :ssh_run "cd %REMOTE_DIR% && test -f .env || cp .env.example .env"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Could not create/verify remote .env. Droplet may be unreachable.
    exit /b 1
)

set "ENV_TMP=%TEMP%\%PROJECT_NAME%-configure-env.sh"
powershell -NoProfile -Command "(Get-Content -Raw '%DEPLOY_DIR%\configure-env.sh') -replace [char]13, '' | Set-Content -NoNewline '%ENV_TMP%' -Encoding ascii"
if not exist "%ENV_TMP%" (
    echo [ERROR] Could not prepare configure-env.sh.
    exit /b 1
)

call :ssh_pipe "%ENV_TMP%" "bash -s -- '%REMOTE_DIR%' '%POSTGRES_USER%' '%POSTGRES_DB%' '%POSTGRES_PASSWORD%' '%WEB_PORT%' '%TAVILY_API_KEY%' '%DOMAIN%'"
set ENV_RC=%ERRORLEVEL%
del "%ENV_TMP%" 2>nul
if not %ENV_RC% equ 0 (
    echo [ERROR] Failed to configure remote .env - SSH error or sed failure.
    echo         Nothing was started. Re-run sync.bat once the droplet is reachable.
    exit /b 1
)
echo [OK] Environment ready.

REM -- Start containers -----------------------------------------------
echo [5/5] Starting containers...
set "COMPOSE_CMD=docker compose --env-file %REMOTE_DIR%/.env -f %REMOTE_DIR%/quick_pp/app/docker/docker-compose.yaml"

set "REMOTE_CMD=!COMPOSE_CMD! up -d"
if %BUILD% equ 1 set "REMOTE_CMD=!REMOTE_CMD! --build"

call :ssh_run "!REMOTE_CMD!"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Remote deploy failed. Check the droplet:
    echo         ssh -i "%KEY_FILE%" root@%PUBLIC_IP%
    exit /b 1
)

REM Always restart qpp-frontend on sync so SvelteKit rebuilds with latest uploaded files
call :ssh_run "!COMPOSE_CMD! restart qpp-frontend" >nul 2>&1

echo.
echo [OK] Sync complete.
if %BUILD% equ 1 echo [OK] Containers rebuilt.

echo.
echo [INFO] Container status:
call :ssh_run "!COMPOSE_CMD! ps"

REM -- Quick tunnel URL (random *.trycloudflare.com) -------------------
set QUICK_URL=
echo.
echo [INFO] Waiting for the Cloudflare Quick Tunnel URL...
for /l %%n in (1,1,15) do (
    if "!QUICK_URL!"=="" (
        ssh %SSH_OPTS% root@%PUBLIC_IP% "!COMPOSE_CMD! logs quicktunnel | grep -o https://[a-z0-9-]*\.trycloudflare\.com | tail -1" <nul > "%TEMP%\%PROJECT_NAME%-quick-url.txt" 2>nul
        for /f "usebackq delims=" %%u in ("%TEMP%\%PROJECT_NAME%-quick-url.txt") do set QUICK_URL=%%u
        if "!QUICK_URL!"=="" ping -n 3 127.0.0.1 >nul 2>&1
    )
)
del "%TEMP%\%PROJECT_NAME%-quick-url.txt" 2>nul

echo.
if not "!QUICK_URL!"=="" (
    echo [OK] App ^(HTTPS Tunnel^): !QUICK_URL!
    echo [INFO] Note: First frontend build takes ~30-60s inside qpp-frontend.
    echo        If you see 502 Bad Gateway initially, refresh after ~30 seconds.
) else (
    echo [WARN] Quick tunnel URL not ready yet. Check:
    echo        ssh -i "%KEY_FILE%" root@%PUBLIC_IP% "!COMPOSE_CMD! logs quicktunnel"
)

goto :sync_end

REM -------------------------------------------------------------------
REM  Retry helpers. Transient SSH failures are common, so retry a few
REM  times before giving up. Both return 0 on success, 1 on final failure.
REM -------------------------------------------------------------------
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

