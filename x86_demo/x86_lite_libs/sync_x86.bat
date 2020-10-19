@echo off
setlocal
setlocal enabledelayedexpansion

rem paddle repo dir
set BASE_REPO_PATH=D:\Paddle-Lite
set PADDLE_LITE_DIR=%BASE_REPO_PATH%\build.lite.x86\inference_lite_lib
echo PADDLE_LITE_DIR is %PADDLE_LITE_DIR%

set LITE_FULL_LIB_NAME=libpaddle_api_full_bundled.lib
set LITE_TINY_LIB_NAME=libpaddle_api_light_bundled.lib
set LITE_MKLML_LIB_NAME=mklml.lib
set LITE_IOMP5_LIB_NAME=libiomp5md.lib
set LITE_MKLML_DLL_NAME=mklml.dll
set LITE_IOMP5_DLL_NAME=libiomp5md.dll

rem paddle full lib
set LITE_FULL_LIB=%PADDLE_LITE_DIR%\cxx\lib\%LITE_FULL_LIB_NAME%
set LITE_TINY_LIB=%PADDLE_LITE_DIR%\cxx\lib\%LITE_TINY_LIB_NAME%

rem paddld include dir
set LITE_INC_DIR=%PADDLE_LITE_DIR%\cxx\include

rem MKLML Lib
set LITE_IOMP5_LIB=%PADDLE_LITE_DIR%\third_party\mklml\lib\%LITE_IOMP5_LIB_NAME%
set LITE_MKLML_LIB=%PADDLE_LITE_DIR%\third_party\mklml\lib\%LITE_MKLML_LIB_NAME%
set LITE_IOMP5_DLL=%PADDLE_LITE_DIR%\third_party\mklml\lib\%LITE_MKLML_DLL_NAME%
set LITE_MKLML_DLL=%PADDLE_LITE_DIR%\third_party\mklml\lib\%LITE_IOMP5_DLL_NAME%

rem target dirs
set target_dir=%~dp0
set target_lib=%target_dir%lib
set target_inc=%target_dir%include

echo ---------------Prepare target dirs-----------------
if EXIST "%target_lib%" (
    call:rm_file_dir "%target_lib%"
)
md "%target_lib%"
echo %target_lib% created
if EXIST "%target_inc%" (
    call:rm_file_dir "%target_inc%"
)
md "%target_inc%"
echo %target_inc% created

echo ---------------COPY Paddle-Lite Full Libs-----------------
echo copy from == %LITE_FULL_LIB%
echo copy to ==== %target_lib%
copy "%LITE_FULL_LIB%" "%target_lib%"

echo ---------------COPY Paddle-Lite Tiny Libs-----------------
echo copy from == %LITE_TINY_LIB%
echo copy to ==== %target_lib%
copy "%LITE_TINY_LIB%" "%target_lib%"

echo ---------------COPY Paddle-Lite Headers-----------------
echo copy from == %LITE_INC_DIR%
echo copy to ==== %target_inc%
xcopy "%LITE_INC_DIR%" "%target_inc%"

echo ---------------COPY MKLML Libs-----------------
echo copy from == %LITE_IOMP5_LIB%
echo copy from == %LITE_MKLML_LIB%
echo copy to ==== %target_lib%
copy %LITE_IOMP5_LIB% %target_lib%
copy %LITE_MKLML_LIB% %target_lib%
copy %LITE_IOMP5_DLL% %target_lib%
copy %LITE_MKLML_DLL% %target_lib%

echo ---------------List Files-----------------
tree /F /A
goto:eof

:rm_file_dir
    del /f /s /q "%~1\*.*"  >nul 2>&1
    rd /s /q  "%~1" >nul 2>&1
goto:eof