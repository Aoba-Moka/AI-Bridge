@echo off
setlocal enabledelayedexpansion
title ComfyUI ^<-^> DiceFrame bridge
chcp 65001 >nul

REM ============================================================
REM  ComfyUI <-> DiceFrame bridge launcher (Windows)
REM
REM  Nothing below has to be edited: the Python interpreter and
REM  the DiceFrame data directory are both auto-detected. If a
REM  guess is wrong, override it with an environment variable.
REM
REM  This file is deliberately ASCII-only, and it contains no
REM  unquoted parentheses inside IF/FOR blocks. cmd ends such a
REM  block at the first ")" and then dies with the famously
REM  unhelpful ". was unexpected at this time." Keep it that way.
REM ============================================================

set "HERE=%~dp0"
cd /d "%HERE%"
set "PYWHY="

REM ------------------------------------------------------------
REM  1) Find a Python that is 3.9 or newer AND has "requests".
REM     Search order:
REM       %CB_PYTHON%  >  py -3 / -3.13 / ...  >  python on PATH
REM                    >  %LOCALAPPDATA%\Programs\Python\Python3*\
REM     Override with:  set "CB_PYTHON=C:\full\path\python.exe"
REM     Note: "py -3" may pick a brand new Python that has no
REM     packages installed, which is why "requests" is checked.
REM ------------------------------------------------------------
set "PY="
if defined CB_PYTHON call :try_py "%CB_PYTHON%"

if not defined PY for %%V in (3 3.13 3.12 3.11 3.10 3.9) do if not defined PY for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do call :try_py "%%P"

REM  "where python" can return the Microsoft Store stub in WindowsApps,
REM  which is not a real interpreter - skip anything pointing there.
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do echo %%P | findstr /i /c:"WindowsApps" >nul || call :try_py "%%P"

if not defined PY for /f "delims=" %%D in ('dir /b /o-n /ad "%LOCALAPPDATA%\Programs\Python" 2^>nul') do call :try_py "%LOCALAPPDATA%\Programs\Python\%%D\python.exe"

if defined PY goto :found_python

echo   [ERROR] No usable Python interpreter found.
echo.
if defined PYWHY goto :python_no_requests
echo   The bridge needs Python 3.9 or newer.
echo   Install it from python.org, ticking "Add python.exe to PATH",
echo   then run this file again. Or set CB_PYTHON to the full path
echo   of your python.exe.
echo.
pause
exit /b 1

:python_no_requests
echo   Found  : %PYWHY%
echo   Problem: the "requests" package is not installed in it.
echo   Fix it : %PYWHY% -m pip install requests
echo.
pause
exit /b 1

:found_python

REM ------------------------------------------------------------
REM  2) Find DiceFrame's data directory (OPTIONAL).
REM     Only used to reuse DiceFrame's own LLM provider as the
REM     Chinese-to-English translator, so no API key has to be
REM     typed here. Override with:
REM       set "CB_DICEFRAME_DATA=D:\wherever\DiceFrame\data"
REM     Not using DiceFrame, or prefer your own translator? Set
REM     the CB_TRANSLATE_* trio below instead.
REM ------------------------------------------------------------
set "DF_DATA=%CB_DICEFRAME_DATA%"
if not defined DF_DATA call :find_diceframe_data
set "CB_TRANSLATE_FROM_DICEFRAME="
if defined DF_DATA if exist "%DF_DATA%\config.json" set "CB_TRANSLATE_FROM_DICEFRAME=%DF_DATA%"

REM  Alternative translator (used instead of the block above).
REM  Any OpenAI-compatible chat endpoint works.
REM set "CB_TRANSLATE_URL=https://api.deepseek.com/v1"
REM set "CB_TRANSLATE_MODEL=deepseek-chat"
REM set "CB_TRANSLATE_KEY=sk-put-your-key-here"
REM
REM  When translation fails the bridge answers 502 instead of
REM  handing DiceFrame a blank image. For the old "draw it anyway"
REM  behaviour, uncomment:
REM set "CB_ON_TRANSLATE_FAILURE=warn"

REM ------------------------------------------------------------
REM  3) Prompt settings.
REM     Type your words BETWEEN the quotes and leave the rest of
REM     the line alone. The quotes are what make commas, colons,
REM     parentheses and ampersands safe here - do not remove them.
REM ------------------------------------------------------------

REM  Fixed prompt: appended to EVERY image. Prefix goes first,
REM  suffix goes last, joined with ", ". Neither is translated,
REM  so write English / danbooru tags.
REM  Example: masterpiece, best quality, cinematic lighting
set "CB_POSITIVE_PREFIX="
set "CB_POSITIVE_SUFFIX="

REM  Negative prompt: what you do NOT want. These are APPENDED to
REM  the built-in list (lowres / bad anatomy / watermark ...),
REM  which stays intact.
REM
REM  WARNING: never put composition words here - frame, borders,
REM  letterboxed, black bars, film strip, out of frame. DiceFrame
REM  appends "no frame" to its avatar prompts, and a negative
REM  "frame" fights it: the character collapses into a small block
REM  on a white canvas.
REM  Example: extra fingers, mutated hands, long neck, off-model
REM set "CB_ANIMA_NEGATIVE_EXTRA=extra fingers, mutated hands, long neck, off-model"
REM set "CB_NEGATIVE_EXTRA="

REM  Want to REPLACE the built-in list instead of appending to it?
REM  Fill one of these in. Rarely what you want - you lose the
REM  built-in quality tags.
REM  Example: lowres, worst quality, bad anatomy, bad hands, watermark
set "CB_ANIMA_NEGATIVE="

REM ------------------------------------------------------------
REM  4) Listen port. Override with CB_PORT.
REM     This must match the Base URL configured in DiceFrame:
REM         http://127.0.0.1:<PORT>/v1
REM     "Connection refused" on the DiceFrame side usually means
REM     something else already holds the port. Check with:
REM         netstat -ano | findstr :8192
REM ------------------------------------------------------------
set "PORT=8192"
if defined CB_PORT set "PORT=%CB_PORT%"

echo.
echo ============================================================
echo   ComfyUI - DiceFrame bridge
echo   python : %PY%
echo   listen : http://127.0.0.1:%PORT%
echo   base   : http://127.0.0.1:%PORT%/v1
echo   model  : anime = Anima  ^|  realistic = SD1.5 checkpoint
echo   usage  : %PY% comfy_bridge.py --help
if not "%CB_PYTHON%"=="" echo   python : pinned by CB_PYTHON
if not "%CB_DICEFRAME_DATA%"=="" echo   df data: !CB_DICEFRAME_DATA!
if not "%CB_TRANSLATE_FROM_DICEFRAME%"=="" echo   translate: reuse DiceFrame LLM provider
if not "%CB_TRANSLATE_URL%"=="" echo   translate: !CB_TRANSLATE_URL!
if not "%CB_POSITIVE_PREFIX%"=="" echo   fixed pre : !CB_POSITIVE_PREFIX!
if not "%CB_POSITIVE_SUFFIX%"=="" echo   fixed post: !CB_POSITIVE_SUFFIX!
if not "%CB_ANIMA_NEGATIVE_EXTRA%"=="" echo   neg extra : !CB_ANIMA_NEGATIVE_EXTRA!
if not "%CB_NEGATIVE_EXTRA%"=="" echo   neg extra : !CB_NEGATIVE_EXTRA!
if not "%CB_ANIMA_NEGATIVE%"=="" echo   neg list  : built-in list REPLACED (see this file)
echo ------------------------------------------------------------
echo   Keep this window open while playing.
echo   Close it (or press Ctrl+C) to stop the bridge.
echo ============================================================
echo.

:loop
"%PY%" "%HERE%comfy_bridge.py" --port %PORT% %*
set "RC=%ERRORLEVEL%"
echo.
echo [bridge exited with code %RC%] restarting in 5 seconds...
echo (press Ctrl+C now to stop)
timeout /t 5 /nobreak >nul
goto loop

REM ============================================================
REM  Subroutines. Everything below is only reached via CALL.
REM ============================================================

:try_py
REM  %~1 = a candidate interpreter path.
REM  Accepts it only if it is Python 3.9+ AND imports requests.
if defined PY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys;sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
"%~1" -c "import requests" >nul 2>nul
if errorlevel 1 set "PYWHY=%~1"
if not errorlevel 1 set "PY=%~1"
exit /b 0

:find_diceframe_data
REM  Shallow scan of likely locations. A candidate counts only if
REM  it has data\config.json AND that file mentions "ai_providers",
REM  so we do not lock onto some other app's config.json.
REM
REM  Each root is passed as an ARGUMENT to :scan_root, never pasted
REM  into a "for ... in (...)" set - a ")" in a path would end the
REM  set early and kill the script.
set "DF_DATA="
call :scan_root "%~d0"
call :scan_root "%HERE%."
call :scan_root "%HERE%.."
call :scan_root "%USERPROFILE%"
call :scan_root "%USERPROFILE%\Desktop"
call :scan_root "%USERPROFILE%\Documents"
call :scan_root "%LOCALAPPDATA%\Programs"
call :scan_root "%ProgramFiles%"
for %%D in (C D E F G) do call :scan_root "%%D:"
REM  One level deeper handles the common "D:\download\DiceFrame-x.y"
REM  layout. Only top-level folders are listed, so this stays cheap.
for %%D in (C D E F G) do call :scan_deep "%%D:"
call :scan_deep "%USERPROFILE%\Downloads"
call :scan_deep "%USERPROFILE%\Desktop"
exit /b 0

:scan_root
REM  %~1 = directory to look in, for a "DiceFrame*" subdirectory.
if defined DF_DATA exit /b 0
if not exist "%~1" exit /b 0
for /f "delims=" %%N in ('dir /b /ad "%~1\DiceFrame*" 2^>nul') do call :check_df_dir "%~1\%%N"
exit /b 0

:scan_deep
REM  %~1 = directory whose immediate subdirectories to scan.
if defined DF_DATA exit /b 0
if not exist "%~1" exit /b 0
for /f "delims=" %%P in ('dir /b /ad "%~1\" 2^>nul') do call :scan_root "%~1\%%P"
exit /b 0

:check_df_dir
REM  %~1 = a candidate DiceFrame install root.
if defined DF_DATA exit /b 0
if not exist "%~1\data\config.json" exit /b 0
findstr /i /c:"ai_providers" "%~1\data\config.json" >nul 2>nul || exit /b 0
set "DF_DATA=%~1\data"
exit /b 0
