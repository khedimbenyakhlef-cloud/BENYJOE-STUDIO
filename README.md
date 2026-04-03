# 🎬🎙️🎵 BENY-JOE CINÉ IA PRO v6.0

> **Studio Génératif IA Professionnel — 100% Gratuit**
> Fondé par **KHEDIM BENYAKHLEF dit BENY-JOE**

## 🆕 Nouveautés v6.0
- ✅ Barre de progression 100% fonctionnelle
- ✅ Génération async réelle (non bloquante)
- ✅ Auto-update URL Render depuis Colab/Kaggle
- ✅ Polling GPU robuste avec retry exponentiel
- ✅ Compatible Colab ET Kaggle

## 🚀 Démarrage rapide

### 1. Colab/Kaggle — GPU
1. Ouvrez `kaggle/KAGGLE_NOTEBOOK_v6.py`
2. Copiez chaque bloc dans une cellule séparée
3. Modifiez `NGROK_TOKEN` et `RENDER_URL` dans la cellule 11
4. Run All → l'URL est envoyée automatiquement à Render!

### 2. Render — Backend
- Build: `pip install -r requirements.txt`
- Start: `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --timeout 420 --workers 1 --threads 4`
- Env: `ACCESS_PIN=2022002`

### 3. Utilisation
- Ouvrez votre URL Render → PIN: `2022002`

---

## 🆕 Nouveautés v10.1 — Lampe statut + Sélecteur CPU/GPU

### Nouveaux endpoints backend
| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/api/engine/status` | Statut lampe (active/inactive) + device actif |
| POST | `/api/engine/device` | Change le device CPU/GPU |
| POST | `/api/engine/generate` | Lance génération locale (asynchrone) |
| POST | `/api/engine/cancel` | Reset d'urgence de la lampe |

### Interface panneau statut
Ouvrir `frontend/status_panel.html` pour le panneau visuel avec :
- 🟢 Lampe verte = moteur actif (génération en cours)
- 🔴 Lampe rouge = moteur inactif
- Toggle CPU / GPU (CUDA)
- Boutons Générer / Reset

### Lancer les tests
```bash
pip install pytest
pytest tests/test_engine.py -v
```

### Créer le méga-zip
```bash
bash build_zip.sh
```
