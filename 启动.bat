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



echo    1) CLI 聊天      16K 上下文 + bf16, 深度 5（短对话最快）



echo    2) API 服务      64K / 128K / 256K 可选, 端口 8099/8100/8101



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



"%MINI%" "%MODEL%" --chat --kv bf16 --ctx 16384 --spec mtp --draft-tokens 5 %EFF% --max-new 8192



chcp 936 >nul



goto done







:longcli



echo.



chcp 65001 >nul



"%MINI%" "%MODEL%" --chat %COMMON% %LONG% --spec mtp --draft-tokens 0 %EFF% --max-new 8192



chcp 936 >nul



goto done







:midcli
echo.
chcp 65001 >nul
"%MINI%" "%MODEL%" --chat --kv rk4v4-e8 --ctx 131072 --spec mtp --draft-tokens 1 %EFF% --think --max-new 8192
chcp 936 >nul
goto done

:api
set "RUNNING="
for /f "tokens=2" %%I in ('tasklist /fi "imagename eq infer-serve.exe" 2^>nul ^| find /i "infer-serve.exe"') do set "RUNNING=%%I"
if defined RUNNING (
  echo [错误] 已经有 infer-serve.exe 在运行（PID %RUNNING%），先停止它再启动，避免两个服务抢显存。
  goto menu
)
echo.
echo [API 服务] 选择上下文档（直接回车 = 1）
echo   1) 64K  + KV 窗口, 深度 5, 端口 8099
echo   2) 128K, 深度 1, 无窗口, 端口 8100
echo   3) 256K + KV 窗口, 深度 1, 端口 8101
set "ASEL="
set /p "ASEL=请选择 [1-3]: "
if "%ASEL%"=="2" goto apimid
if "%ASEL%"=="3" goto apilong
goto apishort

:apishort
echo.
echo [API 服务] http://127.0.0.1:8099/v1   64K, 深度 5   按 Ctrl+C 停止
chcp 65001 >nul
"%SERVE%" "%MODEL%" --host 127.0.0.1 --port 8099 --model-name ninfer-qwen3 --ctx 65536 --resident-tokens 32768 --sink-tokens 2048 --archive-mib 2048 --kv rk4v4-e8 --spec mtp --draft-tokens 5 --effort medium
chcp 936 >nul
goto done

:apimid
echo.
echo [API 服务] http://127.0.0.1:8100/v1   128K, 深度 1   按 Ctrl+C 停止
chcp 65001 >nul
"%SERVE%" "%MODEL%" --host 127.0.0.1 --port 8100 --model-name ninfer-qwen3 --ctx 131072 --kv rk4v4-e8 --spec mtp --draft-tokens 1 --cache-mib 4096 --no-vision --effort medium
chcp 936 >nul
goto done

:apilong
echo.
echo [API 服务] http://127.0.0.1:8101/v1   256K, 深度 1   按 Ctrl+C 停止
chcp 65001 >nul
"%SERVE%" "%MODEL%" --host 127.0.0.1 --port 8101 --model-name ninfer-qwen3 --ctx 262144 --resident-tokens 32768 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1 --kv rk4v4-e8 --spec mtp --draft-tokens 1 --effort medium
chcp 936 >nul
goto done

:done



echo.



echo [引擎已退出]



pause



goto menu