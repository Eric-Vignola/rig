@echo off
rem Builds rig.mll, rig's plug-in (the rigExpression node rig.fuse() compiles into), into
rem <this folder>\<Maya version>\rig.mll: rig loads it from there when neither RIG_PLUGIN
rem nor MAYA_PLUG_IN_PATH gives one (rig._internal.fuse). build.sh is the macOS twin.
rem Usage: build.bat [maya version ...]   e.g. build.bat 2025 2027
rem   No version (a double-click): every Maya installed with its headers, then a pause.
rem
rem The recipe follows MPyNode's native toolchain:
rem   * MSVC 2022 (v143) located through vswhere, set up with vcvarsall x64;
rem   * the source (src\) is copied to a local %TEMP% folder, compiled and linked there, and
rem     only the .mll is copied back: on a streamed drive (Google Drive) the linker's
rem     memory-mapped write never completes and the process cannot be killed. /IMPLIB and /Fo
rem     are pinned to that folder too, so no .lib / .exp / .obj lands here;
rem   * /O2 with an explicit /fp:precise: no FMA contraction or reassociation, so the node
rem     rounds like the Maya nodes it replaces.
setlocal
set "HERE=%~dp0"
set "VERSIONS=%*"
set PAUSE=
set FAILED=
if not "%~1"=="" goto versions
set PAUSE=1
for /d %%d in ("%ProgramFiles%\Autodesk\Maya20??") do if exist "%%d\include\maya\MPxNode.h" call :add "%%~nxd"
if not defined VERSIONS (echo no Maya with its headers in "%ProgramFiles%\Autodesk" 1>&2 & set FAILED=1 & goto done)
:versions

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -version [17.0^,18.0^) -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINST=%%i"
if not defined VSINST (echo Visual Studio 2022 with the C++ tools not found 1>&2 & set FAILED=1 & goto done)
rem vcvarsall runs vswhere.exe by name: its folder on PATH (else "'vswhere.exe' is not recognized")
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSINST%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul || (set FAILED=1 & goto done)

for %%v in (%VERSIONS%) do call :build %%v || set FAILED=1

:done
if defined PAUSE pause
if defined FAILED exit /b 1
exit /b 0

:add
set "NAME=%~1"
set "VERSIONS=%VERSIONS% %NAME:Maya=%"
exit /b 0

:build
set "VER=%~1"
set "MAYA=%ProgramFiles%\Autodesk\Maya%VER%"
if not exist "%MAYA%\include\maya\MPxNode.h" (echo no Maya %VER% headers at "%MAYA%\include" 1>&2 & exit /b 1)
set "TMPB=%TEMP%\rig_mll_build_%VER%_%RANDOM%_%RANDOM%"
mkdir "%TMPB%" || exit /b 1
copy /Y "%HERE%src\rig.cpp" "%TMPB%\rig.cpp" >nul || goto build_failed
copy /Y "%HERE%src\rig_math.h" "%TMPB%\rig_math.h" >nul || goto build_failed
pushd "%TMPB%"
cl /nologo /c /std:c++17 /O2 /fp:precise /EHsc /MD /utf-8 ^
   /D NT_PLUGIN /D REQUIRE_IOSTREAM /D _BOOL /D WIN32 /D _WINDOWS /D _CRT_SECURE_NO_WARNINGS ^
   /I "%MAYA%\include" "%TMPB%\rig.cpp" /Fo"%TMPB%\rig.obj"
if errorlevel 1 (popd & goto build_failed)
link /nologo /DLL /INCREMENTAL:NO /DEBUG:NONE "%TMPB%\rig.obj" ^
   /LIBPATH:"%MAYA%\lib" OpenMaya.lib Foundation.lib ^
   /EXPORT:initializePlugin /EXPORT:uninitializePlugin ^
   /IMPLIB:"%TMPB%\rig.lib" /OUT:"%TMPB%\rig.mll"
if errorlevel 1 (popd & goto build_failed)
popd
if not exist "%HERE%%VER%" mkdir "%HERE%%VER%"
copy /Y "%TMPB%\rig.mll" "%HERE%%VER%\rig.mll" >nul || (echo "%HERE%%VER%\rig.mll" could not be replaced: close the Maya that has it loaded 1>&2 & goto build_failed)
rd /s /q "%TMPB%" 2>nul
echo built %HERE%%VER%\rig.mll (Maya %VER%)
exit /b 0

:build_failed
rd /s /q "%TMPB%" 2>nul
echo the build for Maya %VER% failed 1>&2
exit /b 1
