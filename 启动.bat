@echo off
chcp 936 >nul
setlocal EnableExtensions
title NInfer-HIP 0.1.3r3
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "PATH=C:\Program Files\AMD\ROCm\7.2\bin;%PATH%"

if not exist "%HERE%\model.txt" (
  > "%HERE%\model.txt" echo E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer
)
set /p MODEL=<"%HERE%\model.txt"
set "MINI=%HERE%\bin\infer-mini.exe"
if not exist "%HERE%\effort.txt" ( > "%HERE%\effort.txt" echo auto )
set /p EFFORT=<"%HERE%\effort.txt"
set "SERVE=%HERE%\bin\infer-serve.exe"
if not exist "%MINI%" (
  echo [错误] 找不到 %MINI%
  pause
  exit /b 1
)
set "COMMON=--kv rk4v4-e8"
set "EFF="
if /i not "%EFFORT%"=="auto" set "EFF=--effort %EFFORT%"
set "SEFF=--think"
if /i "%EFFORT%"=="none" set "SEFF=--no-think"
if /i not "%EFFORT%"=="auto" if /i not "%EFFORT%"=="none" set "SEFF=--effort %EFFORT%"
set "LONG=--ctx 262144 --resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1 --archive-mib 4096"

:menu
cls
echo.
echo   ============================================================
echo    NInfer-HIP 0.1.3r3       RX 9070 XT / ROCm 7.2
echo   ============================================================
echo    模型: %MODEL%
echo.
echo    1) CLI 聊天      8K 上下文, MTP 自动（最快）
echo    2) API 服务      262K 上下文 + KV 记忆, 端口 8099
echo    3) CLI 聊天      262K 上下文 + KV 记忆（长文档）
echo    4) 思考等级      当前: %EFFORT%   (auto/low/medium/xhigh/none)
echo    5) 改模型路径
echo    0) 退出
echo.
echo    提示: /exit 退出  /clear 清空  /stats 看速度  /think off 关思考  /effort low^|medium^|xhigh 调等级
echo.
set "P="
set /p "P=请选择 [0-4]: "

if "%P%"=="1" goto cli
if "%P%"=="2" goto api
if "%P%"=="3" goto longcli
if "%P%"=="4" goto editeffort
if "%P%"=="5" goto editmodel
if "%P%"=="0" exit /b 0
goto menu

:cli
echo.
chcp 65001 >nul
"%MINI%" "%MODEL%" --chat %COMMON% --ctx 16384 --spec mtp --draft-tokens 0 %EFF% --max-new 8192
chcp 936 >nul
goto done

:longcli
echo.
chcp 65001 >nul
"%MINI%" "%MODEL%" --chat %COMMON% %LONG% --spec mtp --draft-tokens 0 %EFF% --max-new 8192
chcp 936 >nul
goto done

:api
echo.
echo [API 服务] http://127.0.0.1:8099/v1     按 Ctrl+C 停止
chcp 65001 >nul
"%SERVE%" "%MODEL%" %COMMON% %LONG% --port 8099 %SEFF%
chcp 936 >nul
goto done

:editeffort
echo.
echo 当前思考等级: %EFFORT%
echo   auto   = 模板默认（最深，xhigh）
echo   xhigh  = 最深思考   medium = 中等   low = 轻思考   none = 关闭思考
set "E="
set /p "E=输入等级（直接回车取消）: "
if not "%E%"=="" > "%HERE%\effort.txt" echo %E%
goto menu

:editmodel
echo.
echo 当前: %MODEL%
set "M="
set /p "M=新的模型路径（回车取消）: "
if not "%M%"=="" > "%HERE%\model.txt" echo %M%
goto menu

:done
echo.
echo [引擎已退出]
pause
goto menu