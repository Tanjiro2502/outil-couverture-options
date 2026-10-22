@echo off
rem Lance l'outil sans taper de commande : double-cliquez sur ce fichier.
rem python -m streamlit contourne le blocage de streamlit.exe par Windows.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app.py
pause
