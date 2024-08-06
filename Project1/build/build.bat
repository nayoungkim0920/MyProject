@echo off
rem Set up environment
set BUILD_DIR=build
rem Clean old build directory
if exist %BUILD_DIR% rd /s /q %BUILD_DIR%
mkdir %BUILD_DIR%
cd %BUILD_DIR%
rem Run CMake
cmake -G "Visual Studio 16 2019" ..