@echo off
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Building executable...
pyinstaller --onefile --windowed --name="CTF_Viewer" --icon=NONE ctf.py

echo.
echo Build complete! Check the 'dist' folder for CTF_Viewer.exe
pause

