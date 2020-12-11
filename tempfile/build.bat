@echo off
setlocal
setlocal enabledelayedexpansion

set source_path=%~dp0
set workspace=%source_path%
REM Set Win32 or x64 platform
REM set BUILD_PLATFORM=Win32
set BUILD_PLATFORM=x64
REM Set /MT (ON) or /MD (OFF)
set MSVC_STATIC_CRT=OFF
@REM set WITH_STATIC_MKL=ON
REM set USE_FULL_API=ON
@REM set USE_FULL_API=OFF
@REM set USE_SHARED_API=OFF

@REM REM "Set Paddle-Lite Lib Dir"
@REM call :getabsolute "..\..\inference_lite_lib"
@REM set PADDLE_LITE_DIR=%absolute%

set vcvarsall_dir=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat

echo "------------------------------------------------------------------------------------------------------|"
@REM echo "|  PADDLE_LITE_DIR=%PADDLE_LITE_DIR%                                                                  |"
echo "|  BUILD_PLATFORM=%BUILD_PLATFORM%                                                                    |"
@REM echo "|  WITH_STATIC_MKL=%WITH_STATIC_MKL%                                                                  |"
echo "|  MSVC_STATIC_CRT=%MSVC_STATIC_CRT%                                                                  |"
@REM echo "|  USE_FULL_API=%USE_FULL_API%                                                                        |"
@REM echo "|  USE_SHARED_API=%USE_SHARED_API%                                                                    |"
echo "------------------------------------------------------------------------------------------------------|"

set build_directory=%workspace%\build
REM "Clean the build directory."
if EXIST "%build_directory%" (
    call:rm_rebuild_dir "%build_directory%"
)

md "%build_directory%"
cd "%build_directory%"
cmake %workspace%  -G "Visual Studio 14 2015" -A %BUILD_PLATFORM% ^
            -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT%
            @REM -DWITH_STATIC_MKL=%WITH_STATIC_MKL% ^
            @REM -DUSE_SHARED_API=%USE_SHARED_API% ^
            @REM -DPADDLE_LITE_DIR=%PADDLE_LITE_DIR% ^
            @REM -DUSE_FULL_API=%USE_FULL_API%

if "%BUILD_PLATFORM%"=="x64" (
    call "%vcvarsall_dir%" amd64
    msbuild /maxcpucount /p:Configuration=Release main.vcxproj
) else (
    call "%vcvarsall_dir%" x86
    msbuild /maxcpucount /p:Configuration=Release main.vcxproj
)
goto:eof

:rm_rebuild_dir
    del /f /s /q "%~1\*.*"  >nul 2>&1
    rd /s /q  "%~1" >nul 2>&1
goto:eof

:getabsolute
set absolute=%~f1
goto :eof
