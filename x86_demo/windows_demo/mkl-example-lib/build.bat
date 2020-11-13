@echo off
setlocal
setlocal enabledelayedexpansion

set source_path=%~dp0

set vcvarsall_dir=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat
set MKL_ROOT=D:\liqi27\compilers_and_libraries_2019.1.144\windows\mkl
set IOMP_ROOT=D:\liqi27\compilers_and_libraries_2019.1.144\windows\compiler
set IOMP_LINK=D:\liqi27\compilers_and_libraries_2019.1.144\windows\redist\intel64\compiler

set build_directory=%source_path%\build
REM "Clean the build directory."
if EXIST "%build_directory%" (
    call:rm_rebuild_dir "%build_directory%"
)

rem "change to build with static / shared"
set LINK_STATIC=OFF

if "%LINK_STATIC%"=="ON" (
    set build_directory=%build_directory%.Static
) else (
    set build_directory=%build_directory%.Shared
)

md "%build_directory%"
cd "%build_directory%"

if "%LINK_STATIC%" == "ON" (
    cmake %source_path%  -G "Visual Studio 14 2015" -A x64 ^
                     -DMKL_ROOT=%MKL_ROOT% ^
                     -DIOMP_ROOT=%IOMP_ROOT% ^
                     -DIOMP_LINK=%IOMP_LINK% ^
                     -DLINK_STATIC=%LINK_STATIC%
 ) else (
    cmake %source_path%  -G "Visual Studio 14 2015" -A x64 ^
                     -DMKL_ROOT=%MKL_ROOT% ^
                     -DIOMP_ROOT=%IOMP_ROOT% ^
                     -DIOMP_LINK=%IOMP_LINK% ^
                     -DLINK_STATIC=%LINK_STATIC% ^
                     -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ^
                     -DBUILD_SHARED_LIBS=TRUE
)


set UCRTVersion=10.0.18362.0
call "%vcvarsall_dir%" amd64

msbuild /maxcpucount:4 /p:Configuration=Release /p:Platform=x64 main_test.vcxproj

goto:eof

:rm_rebuild_dir
    del /f /s /q "%~1\*.*"  >nul 2>&1
    rd /s /q  "%~1" >nul 2>&1
goto:eof