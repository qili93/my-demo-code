@echo off
setlocal
setlocal enabledelayedexpansion

set source_path=%~dp0

set vcvarsall_dir=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat
set MKL_ROOT=D:\liqi27\compilers_and_libraries_2019.1.144\windows\mkl
set IOMP_ROOT=D:\liqi27\compilers_and_libraries_2019.1.144\windows\compiler

set DLL_COPY_DIR=D:\liqi27\compilers_and_libraries_2019.1.144\windows\redist\intel64

set build_directory=%source_path%\build

rem "change to build with static / shared"
set LINK_STATIC=OFF
set WITH_STATIC_MKL=OFF

if "%LINK_STATIC%"=="ON" (
    set build_directory=%build_directory%.Static
) else (
    set build_directory=%build_directory%.Shared
)

if "%WITH_STATIC_MKL%"=="ON" (
    set build_directory=%build_directory%.MKL_LIB
) else (
    set build_directory=%build_directory%.MKL_DLL
)

REM "Clean the build directory."
if EXIST "%build_directory%" (
    call:rm_rebuild_dir "%build_directory%"
)
md "%build_directory%"
cd "%build_directory%"

if "%LINK_STATIC%" == "ON" (
    cmake %source_path%  -G "Visual Studio 14 2015" -A x64 ^
                     -DMKL_ROOT=%MKL_ROOT% ^
                     -DIOMP_ROOT=%IOMP_ROOT% ^
                     -DDLL_COPY_DIR=%DLL_COPY_DIR% ^
                     -DLINK_STATIC=%LINK_STATIC% ^
                     -DWITH_STATIC_MKL=%WITH_STATIC_MKL%
 ) else (
    cmake %source_path%  -G "Visual Studio 14 2015" -A x64 ^
                     -DMKL_ROOT=%MKL_ROOT% ^
                     -DIOMP_ROOT=%IOMP_ROOT% ^
                     -DDLL_COPY_DIR=%DLL_COPY_DIR% ^
                     -DLINK_STATIC=%LINK_STATIC% ^
                     -DWITH_STATIC_MKL=%WITH_STATIC_MKL% ^
                     -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ^
                     -DBUILD_SHARED_LIBS=TRUE
)


set UCRTVersion=10.0.10240.0
call "%vcvarsall_dir%" amd64

msbuild /maxcpucount:4 /p:Configuration=Release /p:Platform=x64 main_test.vcxproj

goto:eof

:rm_rebuild_dir
    del /f /s /q "%~1\*.*"  >nul 2>&1
    rd /s /q  "%~1" >nul 2>&1
goto:eof