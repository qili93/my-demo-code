@echo off
setlocal
setlocal enabledelayedexpansion

rem set BASE_REPO_PATH=C:\Users\liqi27\Downloads\Paddle-Lite
rem set PADDLE_LITE_DIR=%BASE_REPO_PATH%\build.lite.x86\inference_lite_lib
rem echo "------------PADDLE_LITE_DIR is %PADDLE_LITE_DIR%------------"
rem set LD_LIBRARY_PATH=%PADDLE_LITE_DIR%\cxx\lib:%PADDLE_LITE_DIR%\third_party\mklml\lib:%LD_LIBRARY_PATH%cho "------------PADDLE_LITE_DIR is %PADDLE_LITE_DIR%------------"

call :getabsolute "..\..\x86_lite_libs"
set PADDLE_LITE_DIR=%absolute%
echo "------------PADDLE_LITE_DIR is %PADDLE_LITE_DIR%------------"
set LD_LIBRARY_PATH=%PADDLE_LITE_DIR%\lib:%LD_LIBRARY_PATH%

@REM set model dir
call :getabsolute "..\assets\models"
set MODEL_DIR=%absolute%
echo "------------MODEL_DIR is %MODEL_DIR%------------"
set MODEL_TYPE=1 # 0 uncombined; 1 combined paddle fluid model

set MODEL_NAME=pc-seg-float-model

set GLOG_v=5
build\Release\model_test.exe %MODEL_DIR% %MODEL_NAME% %MODEL_TYPE%
goto:eof

:getabsolute
set absolute=%~f1
goto :eof