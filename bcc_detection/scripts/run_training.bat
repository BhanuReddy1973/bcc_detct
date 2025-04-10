@echo off
echo Setting up Python environment...
python -m venv venv
call venv\Scripts\activate

echo Installing requirements...
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir numpy pandas scikit-learn matplotlib seaborn tqdm Pillow

echo Starting training...
python main.py

echo Training completed!
pause 