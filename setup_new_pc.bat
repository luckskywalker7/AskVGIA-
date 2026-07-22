@echo off
REM ============================================================================
REM  setup_new_pc.bat
REM  One-shot environment setup for the Agentic Video Search project on a
REM  fresh Windows PC. Safe to re-run (each step is skipped if already done).
REM
REM  Installs: Python 3.11, ffmpeg, Ollama, a project virtualenv (.venv) with
REM  every dependency pinned to the exact versions in requirements.txt, then
REM  pulls the Ollama models this project expects.
REM
REM  Run this from the project root (double-click it, or `setup_new_pc.bat`
REM  from a cmd prompt).
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Agentic Video Search - New PC Setup
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM 0. winget sanity check (used to auto-install Python/ffmpeg/Ollama)
REM ---------------------------------------------------------------
where winget >nul 2>&1
if errorlevel 1 (
    echo [WARN] winget not found. Python/ffmpeg/Ollama must be installed manually.
    set HAVE_WINGET=0
) else (
    set HAVE_WINGET=1
)

REM ---------------------------------------------------------------
REM 1. Python 3.11
REM ---------------------------------------------------------------
echo [1/6] Checking Python 3.11 ...
py -3.11 -c "print(1)" >nul 2>&1
if errorlevel 1 (
    echo        Not found.
    if "!HAVE_WINGET!"=="1" (
        echo        Installing via winget ...
        winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
    ) else (
        echo [ERROR] Install Python 3.11 manually from https://www.python.org/downloads/ then re-run this script.
        goto :end
    )
) else (
    echo        OK
)

REM ---------------------------------------------------------------
REM 2. ffmpeg (required by moviepy / whisperx for audio extraction)
REM ---------------------------------------------------------------
echo [2/6] Checking ffmpeg ...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo        Not found.
    if "!HAVE_WINGET!"=="1" (
        echo        Installing via winget ...
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        echo        NOTE: open a new terminal after setup so the updated PATH is picked up.
    ) else (
        echo [ERROR] Install ffmpeg manually and add it to PATH: https://www.gyan.dev/ffmpeg/builds/
        goto :end
    )
) else (
    echo        OK
)

REM ---------------------------------------------------------------
REM 3. Ollama (local LLM / VLM backend)
REM ---------------------------------------------------------------
echo [3/6] Checking Ollama ...
where ollama >nul 2>&1
if errorlevel 1 (
    echo        Not found.
    if "!HAVE_WINGET!"=="1" (
        echo        Installing via winget ...
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        echo        NOTE: open a new terminal after setup so the updated PATH is picked up.
    ) else (
        echo [ERROR] Install Ollama manually from https://ollama.com/download then re-run this script.
        goto :end
    )
) else (
    echo        OK
)

REM ---------------------------------------------------------------
REM 4. Project virtualenv + pinned dependencies
REM ---------------------------------------------------------------
echo [4/6] Creating virtualenv (.venv) ...
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
) else (
    echo        Already exists, skipping.
)

echo        Installing pinned dependencies from requirements.txt ...
echo        (torch/torchaudio/torchvision are CUDA cu128 builds - this step
echo         downloads several GB and can take a while)
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed - see output above.
    echo         If this PC has no NVIDIA GPU, edit requirements.txt and drop the
    echo         "+cu128" suffix from the torch/torchaudio/torchvision lines, then re-run.
    goto :end
)

REM ---------------------------------------------------------------
REM 5. .env file + data directories
REM ---------------------------------------------------------------
echo [5/6] Preparing configuration and data folders ...
if not exist ".env" (
    copy /y ".env.example" ".env" >nul
    echo        Created .env from .env.example.
    echo        [ACTION REQUIRED] Edit .env and set HF_TOKEN to your own
    echo        HuggingFace token (https://huggingface.co/settings/tokens) -
    echo        needed for pyannote speaker diarization.
) else (
    echo        .env already exists, leaving it untouched.
)

for %%D in (
    "data\faiss_store"
    "data\graph_store"
    "data\keyframes"
    "data\vision_descriptions"
    "data\audio"
    "data\videos"
    "data\uploads"
) do (
    if not exist "%%~D" mkdir "%%~D"
)

REM ---------------------------------------------------------------
REM 6. Ollama models (matches OLLAMA_CHAT_MODEL / OLLAMA_VISION_MODEL in .env)
REM ---------------------------------------------------------------
echo [6/6] Pulling required Ollama models ...
ollama list >nul 2>&1
if errorlevel 1 (
    echo        Starting Ollama server ...
    start "ollama serve" /min ollama serve
    timeout /t 5 /nobreak >nul
)
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b

echo.
echo ============================================================
echo  Setup complete.
echo ============================================================
echo  Next steps:
echo    1. If .env was just created, set HF_TOKEN in it.
echo    2. Make sure "ollama serve" is running.
echo    3. Start the app:
echo         .venv\Scripts\activate
echo         uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
echo    4. Open http://localhost:8000/app
echo ============================================================

:end
endlocal
pause
