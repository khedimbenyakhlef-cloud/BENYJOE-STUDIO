#!/bin/bash
# ═══════════════════════════════════════════════
# BENY-JOE CINÉ IA PRO v5.0 — Lancement Linux
# ═══════════════════════════════════════════════

echo "🎬 BENY-JOE CINÉ IA PRO v5.0"
echo "==============================="

# Créer .env si absent
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  .env créé depuis .env.example — configurez votre URL ngrok!"
fi

# Installer les dépendances
echo "📦 Installation des dépendances..."
pip install -r requirements.txt -q

# Créer les dossiers nécessaires
mkdir -p outputs logs

# Lancer le serveur
echo "🚀 Démarrage du serveur sur http://localhost:5000"
cd backend && python app.py
