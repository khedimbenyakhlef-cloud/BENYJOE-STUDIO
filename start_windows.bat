@echo off
echo 🎬 BENY-JOE CINE IA PRO v5.0
echo ===============================

if not exist ".env" (
    copy .env.example .env
    echo [!] .env cree - configurez votre URL ngrok
)

echo [*] Installation des dependances...
pip install -r requirements.txt -q

mkdir outputs 2>nul
mkdir logs 2>nul

echo [*] Demarrage serveur sur http://localhost:5000
cd backend && python app.py
pause
