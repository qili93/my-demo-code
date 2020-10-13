@echo off
setlocal
setlocal enabledelayedexpansion

set source_path=%~dp0

set build_directory=%source_path%\build

if EXIST "%build_directory%" (
    call:rm_rebuild_dir "%build_directory%"
) 

md "%build_directory%"
set vcvarsall_dir=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat

IF NOT EXIST "%vcvarsall_dir%" (
  goto set_vcvarsall_dir
) else (
  goto cmake
)

:set_vcvarsall_dir
SET /P vcvarsall_dir="Please input the path of visual studio command Prompt, such as C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat   =======>"
set tmp_var=!vcvarsall_dir!
call:remove_space
set vcvarsall_dir=!tmp_var!   
IF NOT EXIST "!vcvarsall_dir!" (
    echo "------------!vcvarsall_dir! not exist------------"
    goto set_vcvarsall_dir
)

:cmake
cd "%build_directory%"

set BASE_REPO_PATH=C:\Users\liqi27\Downloads\Paddle-Lite
set PADDLE_LITE_DIR=%BASE_REPO_PATH%\build.lite.x86\inference_lite_lib
echo "------------PADDLE_LITE_DIR is %PADDLE_LITE_DIR%------------"
set USE_FULL_API=TRUE

cmake ..   -G "Visual Studio 14 2015 Win64" -T host=x64 ^
                -DPADDLE_LITE_DIR=%PADDLE_LITE_DIR%   ^
                -DUSE_FULL_API=%USE_FULL_API%

call "%vcvarsall_dir%" amd64

msbuild /maxcpucount /p:Configuration=Release  human_seg_demo.vcxproj

goto:eof

:rm_rebuild_dir
    del /f /s /q "%~1\*.*"  >nul 2>&1
    rd /s /q  "%~1" >nul 2>&1
goto:eof

:remove_space
:remove_left_space
if "%tmp_var:~0,1%"==" " (
    set "tmp_var=%tmp_var:~1%"
    goto remove_left_space
)

:remove_right_space
if "%tmp_var:~-1%"==" " (
    set "tmp_var=%tmp_var:~0,-1%"
    goto remove_left_space
)
goto:eof
