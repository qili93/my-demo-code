@echo off
setlocal
setlocal enabledelayedexpansion

REM "Set Paddle-Lite Lib Dir"
call :getabsolute "..\lite_lib_win32_MD_MKL_LIB"
set PADDLE_LITE_DIR=%absolute%
REM set LD_LIBRARY_PATH=%PADDLE_LITE_DIR%\lib:%LD_LIBRARY_PATH%

@REM set model dir
call :getabsolute "..\models"
set MODEL_DIR=%absolute%
set MODEL_NAME=pc-seg-float-model

echo "------------------------------------------------------------------------------------------------------|"
echo "|  PADDLE_LITE_DIR=%PADDLE_LITE_DIR%                                                                  |"
echo "|  MODEL_PATH=%MODEL_DIR%\%MODEL_NAME%                                                                |"
echo "------------------------------------------------------------------------------------------------------|"

set GLOG_v=5
build\Release\model_test.exe %MODEL_DIR%\%MODEL_NAME%
goto:eof

:getabsolute
set absolute=%~f1
goto :eof