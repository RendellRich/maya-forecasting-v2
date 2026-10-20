@echo off
cd /d "%~dp0"
python -m streamlit run app\streamlit_app.py
pause
