@echo off

set FROMFOLDER=%~dp0
set FROMFOLDER=%FROMFOLDER:~0,-1%
for %%i in (%FROMFOLDER%\..) do set FROMFOLDER=%%~fi

set TOFOLDER=%USERPROFILE%\OneDrive\Desktop\VsCode\cpp\my\tools\NeuralNetwork

if not exist %TOFOLDER% (mkdir %TOFOLDER%)

copy %~dp0export.bat %TOFOLDER%

copy %FROMFOLDER%\include\NeuralNetwork.hpp %TOFOLDER%
copy %FROMFOLDER%\src\NeuralNetwork.cpp %TOFOLDER%