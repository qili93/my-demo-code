@echo off
setlocal
setlocal enabledelayedexpansion

@REM REM "Set Paddle-Lite Lib Dir"
@REM call :getabsolute "..\..\inference_lite_lib"
@REM set PADDLE_LITE_DIR=%absolute%
@REM REM set LD_LIBRARY_PATH=%PADDLE_LITE_DIR%\lib:%LD_LIBRARY_PATH%

@REM @REM set model dir
@REM call :getabsolute "..\assets\models"
@REM set MODEL_DIR=%absolute%
@REM set MODEL_NAME=pc-seg-float-model

@REM echo "------------------------------------------------------------------------------------------------------|"
@REM echo "|  PADDLE_LITE_DIR=%PADDLE_LITE_DIR%                                                                  |"
@REM echo "|  MODEL_PATH=%MODEL_DIR%\%MODEL_NAME%                                                                |"
@REM echo "------------------------------------------------------------------------------------------------------|"

@REM set GLOG_v=5
build\Release\model_test.exe
goto:eof

@REM :getabsolute
@REM set absolute=%~f1
@REM goto :eof