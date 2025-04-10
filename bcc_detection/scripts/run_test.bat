@echo off
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

echo Running pipeline test...
python -m scripts.test_pipeline

echo Test completed!
pause 