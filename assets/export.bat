@echo off

set "dir=%*"
set "dir=%dir:"=%"

if not exist "%dir%" (
  echo Export directory doesn't exist %dir%
  exit /b 1
)

set PARENTPATH=%~dp0
if "%PARENTPATH:~-1%"=="\" set "PARENTPATH=%PARENTPATH:~0, -1%"

if not exist "%dir%\include" (
  mkdir "%dir%\include"
  echo added include folder
)
if not exist "%dir%\src" (
  mkdir "%dir%\src"
  echo added src folder
)

copy %FROMFOLDER%\NeuralNetwork.hpp "%dir%\include\NeuralNetwork.hpp"
copy %FROMFOLDER%\NeuralNetwork.cpp "%dir%\src\NeuralNetwork.cpp"

echo Successfully exported NeuralNetwork Tool to %dir%
exit /b 0