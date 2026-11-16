@echo off
rem Builds rig.mll, rig's plug-in (the rigExpression node rig.fuse() compiles into), for one
rem Maya version.
rem Usage: build.bat [maya version] [out dir]   defaults: 2025, .\build\<version>
rem
rem The recipe follows MPyNode's native toolchain:
rem   * MSVC 2022 (v143) located through vswhere, set up with vcvarsall x64;
rem   * the source is copied to a local %TEMP% folder, compiled and linked there, and only
rem     the .mll is copied back: on a streamed drive (Google Drive) the linker's memory-mapped
rem     write never completes and the process cannot be killed. /IMPLIB and /Fo are pinned to
rem     that folder too, so no .lib / .exp / .obj lands here;
rem   * /O2 with an explicit /fp:precise: no FMA contraction or reassociation, so the node
rem     rounds like the Maya nodes it replaces.
setlocal
set HERE=%~dp0
set VER=%~1
if "%VER%"=="" set VER=2025
set OUT=%~2
if "%OUT%"=="" set OUT=%HERE%build\%VER%
set "MAYA=C:\Program Files\Autodesk\Maya%VER%"
if not exist "%MAYA%\include\maya\MPxNode.h" (echo no Maya %VER% headers at "%MAYA%\include" 1>&2 & exit /b 1)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -version [17.0^,18.0^) -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINST=%%i"
if not defined VSINST (echo Visual Studio 2022 with the C++ tools not found 1>&2 & exit /b 1)
call "%VSINST%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul || exit /b 1

set "TMPB=%TEMP%\rig_mll_build_%RANDOM%_%RANDOM%"
mkdir "%TMPB%" || exit /b 1
copy /Y "%HERE%rig.cpp" "%TMPB%\rig.cpp" >nul || exit /b 1
copy /Y "%HERE%rig_math.h" "%TMPB%\rig_math.h" >nul || exit /b 1
pushd "%TMPB%"
cl /nologo /c /std:c++17 /O2 /fp:precise /EHsc /MD /utf-8 ^
   /D NT_PLUGIN /D REQUIRE_IOSTREAM /D _BOOL /D WIN32 /D _WINDOWS /D _CRT_SECURE_NO_WARNINGS ^
   /I "%MAYA%\include" "%TMPB%\rig.cpp" /Fo"%TMPB%\rig.obj"
if errorlevel 1 goto fail
link /nologo /DLL /INCREMENTAL:NO /DEBUG:NONE "%TMPB%\rig.obj" ^
   /LIBPATH:"%MAYA%\lib" OpenMaya.lib Foundation.lib ^
   /EXPORT:initializePlugin /EXPORT:uninitializePlugin ^
   /IMPLIB:"%TMPB%\rig.lib" /OUT:"%TMPB%\rig.mll"
if errorlevel 1 goto fail
popd
if not exist "%OUT%" mkdir "%OUT%"
copy /Y "%TMPB%\rig.mll" "%OUT%\rig.mll" >nul || exit /b 1
rd /s /q "%TMPB%" 2>nul
echo built %OUT%\rig.mll (Maya %VER%)
exit /b 0
:fail
popd
rd /s /q "%TMPB%" 2>nul
exit /b 1
