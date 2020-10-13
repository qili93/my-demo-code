@echo off
setlocal
setlocal enabledelayedexpansion

set BASE_REPO_PATH=C:\Users\liqi27\Downloads\Paddle-Lite
set PADDLE_LITE_DIR=%BASE_REPO_PATH%\build.lite.x86\inference_lite_lib
echo "------------PADDLE_LITE_DIR is %PADDLE_LITE_DIR%------------"

set source_path=%~dp0

set LD_LIBRARY_PATH=%PADDLE_LITE_DIR%\cxx\lib:%PADDLE_LITE_DIR%\third_party\mklml\lib:%LD_LIBRARY_PATH%

rem set model dir
set MODEL_DIR=%source_path%..\assets\models
set MODEL_TYPE=1 # 0 uncombined; 1 combined paddle fluid model

rem set MODEL_NAME=align150-fp32
rem set MODEL_NAME=angle-fp32
rem set MODEL_NAME=detect_rgb-fp32
rem set MODEL_NAME=detect_rgb-int8
rem set MODEL_NAME=eyes_position-fp32
rem set MODEL_NAME=iris_position-fp32
rem set MODEL_NAME=mouth_position-fp32
rem set MODEL_NAME=seg-model-int8
set MODEL_NAME=pc-seg-float-model

set GLOG_v=0
build\Release\human_seg_demo.exe %MODEL_DIR%\%MODEL_NAME% %MODEL_TYPE%

goto:eof