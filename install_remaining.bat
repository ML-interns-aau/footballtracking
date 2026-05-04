@echo off
echo ============================================
echo  Step 1: Install packages that don't pull torch
echo ============================================
venv\Scripts\pip.exe install ^
    streamlit ^
    opencv-python-headless ^
    numpy ^
    tqdm ^
    supervision ^
    scikit-learn ^
    matplotlib ^
    requests ^
    scipy ^
    pandas ^
    plotly

echo.
echo ============================================
echo  Step 2: Install ultralytics without overwriting torch
echo ============================================
venv\Scripts\pip.exe install ultralytics --no-deps
venv\Scripts\pip.exe install "lapx>=0.5.2" "py-cpuinfo" "psutil" "seaborn>=0.11.0" "thop>=0.1.1" "ultralytics-thop>=2.0.0"

echo.
echo ============================================
echo  Step 3: Verify everything
echo ============================================
venv\Scripts\python.exe -c ^
    "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); import streamlit; print('streamlit OK'); import cv2; print('opencv OK'); import ultralytics; print('ultralytics OK')"

echo.
echo ============================================
echo  All done! Run the app with:
echo  venv\Scripts\streamlit.exe run dashboard/Home.py
echo ============================================
pause
