@echo off
setlocal EnableExtensions EnableDelayedExpansion
title NInfer-HIP chat

rem ---------------------------------------------------------------
rem  NInfer-HIP interactive chat launcher
rem  Double-click this file. Put ninfer_mini_nordc.exe next to it
rem  or in .\bin\, and set the model path once (it is remembered
rem  in model.txt).
rem ---------------------------------------------------------------

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"

rem  engine name: current releases ship infer-mini.exe; older ones used
rem  ninfer_mini_nordc.exe. Accept both, prefer the current one.
set "EXE="
for %%N in (infer-mini.exe ninfer_mini_nordc.exe infer_mini_nordc.exe) do (
  if not defined EXE if exist "%HERE%\%%N" set "EXE=%HERE%\%%N"
  if not defined EXE if exist "%HERE%\bin\%%N" set "EXE=%HERE%\bin\%%N"
)
if not defined EXE (
  echo [x] cannot find infer-mini.exe next to this script.
  echo     put the engine next to chat.bat, or in a bin\ subfolder.
  pause
  exit /b 1
)

set "ROCM=C:\Program Files\AMD\ROCm\7.2\bin"
if exist "%ROCM%" set "PATH=%PATH%;%ROCM%"

rem ---- model path: argument > model.txt > NINFER_MODEL > default ----
set "MODEL=%~1"
if not defined MODEL if exist "%HERE%\model.txt" set /p MODEL=<"%HERE%\model.txt"
if not defined MODEL if defined NINFER_MODEL set "MODEL=%NINFER_MODEL%"
if not defined MODEL set "MODEL=E:\infer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer"

if not exist "%MODEL%" (
  echo [i] model artifact not found:
  echo     %MODEL%
  echo.
  echo     drag the .ninfer file into this window and press Enter,
  echo     or download the artifact listed in the release notes.
  set /p "MODEL=path: "
  set "MODEL=!MODEL:"=!"
)
if not exist "%MODEL%" (
  echo [x] still not found, aborting.
  pause
  exit /b 1
)
>"%HERE%\model.txt" echo %MODEL%

:menu
cls
echo ================================================================
echo   NInfer-HIP  -  Qwen3.8-27B on RX 9070 XT 16GB  (RDNA4 / gfx1201)
echo ================================================================
echo   model: %MODEL%
echo.
echo   [1]  daily chat         ctx 16384   + MTP dw2   ~ 78 t/s
echo   [2]  long doc / code   ctx 32768   + MTP dw2   ~ 64 t/s
echo   [3]  long doc / code   ctx 65536   + MTP dw2   ~ 42 t/s
echo   [4]  whole book        ctx 131072  no MTP      ~ 23 t/s
echo   [5]  custom arguments
echo   [q]  quit
echo.
echo   rule: --ctx must stay close to the real prompt length; every
echo         4096 extra tokens of --ctx costs VRAM residency budget.
echo   in chat the history is re-sent each turn, so a long session
echo         fills the same window as a long prompt.
echo ================================================================
set "CH="
set /a TRIES+=1
set /p "CH=choose: "
if not defined CH (
  if !TRIES! GEQ 3 exit /b 0
  goto menu
)
if /i "%CH%"=="1" set "ARGS=--ctx 16384 --kv rk4v4-e8 --spec mtp --draft-tokens 2 --max-new 512"
if /i "%CH%"=="2" set "ARGS=--ctx 32768 --kv rk4v4-e8 --spec mtp --draft-tokens 2 --max-new 512"
if /i "%CH%"=="3" set "ARGS=--ctx 65536 --kv rk4v4-e8 --spec mtp --draft-tokens 2 --max-new 512"
if /i "%CH%"=="4" set "ARGS=--ctx 131072 --kv rk4v4-e8 --max-new 512"
if /i "%CH%"=="5" goto custom
if /i "%CH%"=="q" exit /b 0
if not defined ARGS goto menu

:run
echo.
for %%F in ("%EXE%") do echo   engine: %%~nxF  %%~zF bytes  %%~tF
echo.
"%EXE%" "%MODEL%" --chat %ARGS%
echo.
echo [chat ended]
pause
exit /b 0

:custom
echo.
echo example: --ctx 32768 --kv rk4v4-e8 --spec mtp --draft-tokens 2 --max-new 512 --think
set "ARGS="
set /p "ARGS=arguments: "
if not defined ARGS goto menu
goto run
