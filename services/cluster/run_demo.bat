@echo off
echo Starting CLASP P2 Cluster Service Demonstration...
echo.

cd src
C:\Users\pras2\AppData\Local\Python\pythoncore-3.11-64\python.exe -X utf8 -m cluster.main 2>nul

echo.
echo Demonstration complete.
pause
