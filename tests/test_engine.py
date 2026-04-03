"""
BENY-JOE STUDIO v10.1 — Tests automatisés
Couvre: /api/engine/status, /api/engine/device, /api/engine/generate, /api/engine/cancel
Usage: pytest tests/test_engine.py -v
"""

import pytest
import time
import sys
import os

# Assure que le backend est importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import conditionnel pour éviter erreur si PyTorch absent
try:
    import app as studio_app
    BACKEND_AVAILABLE = True
except Exception as e:
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    if not BACKEND_AVAILABLE:
        pytest.skip(f"Backend non importable: {IMPORT_ERROR}")
    studio_app.app.config['TESTING'] = True
    with studio_app.app.test_client() as c:
        # Reset engine state avant chaque test
        with studio_app._engine_lock:
            studio_app._engine_state['is_generating'] = False
            studio_app._engine_state['device'] = 'cpu'
            studio_app._engine_state['current_job'] = None
        yield c


# ── Tests /api/engine/status ──────────────────────────────────────────────────

class TestEngineStatus:
    def test_status_initial_inactif(self, client):
        """La lampe est rouge (inactive) au démarrage."""
        res = client.get('/api/engine/status')
        assert res.status_code == 200
        data = res.get_json()
        assert 'active' in data
        assert data['active'] is False

    def test_status_device_initial_cpu(self, client):
        """Le device initial est CPU."""
        res = client.get('/api/engine/status')
        data = res.get_json()
        assert data['device'] == 'cpu'

    def test_status_contient_champs_requis(self, client):
        """La réponse contient tous les champs attendus."""
        res = client.get('/api/engine/status')
        data = res.get_json()
        for champ in ('active', 'device', 'job', 'started_at'):
            assert champ in data, f"Champ manquant: {champ}"

    def test_status_job_null_initial(self, client):
        """Aucun job en cours initialement."""
        res = client.get('/api/engine/status')
        data = res.get_json()
        assert data['job'] is None


# ── Tests /api/engine/device ──────────────────────────────────────────────────

class TestEngineDevice:
    def test_set_device_cpu(self, client):
        """Sélection CPU acceptée."""
        res = client.post('/api/engine/device',
                          json={'device': 'cpu'},
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert data.get('ok') is True
        assert data.get('device') == 'cpu'

    def test_set_device_invalide(self, client):
        """Device invalide retourne 400."""
        res = client.post('/api/engine/device',
                          json={'device': 'tpu'},
                          content_type='application/json')
        assert res.status_code == 400
        data = res.get_json()
        assert 'error' in data

    def test_set_device_persiste(self, client):
        """Le changement de device est reflété dans /api/engine/status."""
        client.post('/api/engine/device',
                    json={'device': 'cpu'},
                    content_type='application/json')
        res = client.get('/api/engine/status')
        data = res.get_json()
        assert data['device'] == 'cpu'

    def test_set_device_cuda_sans_gpu(self, client, monkeypatch):
        """
        Si PyTorch ne détecte pas de GPU, la sélection CUDA renvoie 400.
        On mock torch.cuda.is_available() → False.
        """
        import unittest.mock as mock
        try:
            import torch
            with mock.patch('torch.cuda.is_available', return_value=False):
                res = client.post('/api/engine/device',
                                  json={'device': 'cuda'},
                                  content_type='application/json')
                # Doit renvoyer 400 si pas de GPU
                assert res.status_code == 400
        except ImportError:
            pytest.skip("PyTorch non installé")


# ── Tests /api/engine/generate ────────────────────────────────────────────────

class TestEngineGenerate:
    def test_generate_accepte_prompt(self, client):
        """Une génération avec prompt valide retourne 202."""
        res = client.post('/api/engine/generate',
                          json={'prompt': 'Scène cinématographique épique', 'device': 'cpu'},
                          content_type='application/json')
        assert res.status_code == 202
        data = res.get_json()
        assert data.get('accepted') is True
        assert 'job_id' in data

    def test_generate_sans_prompt(self, client):
        """Sans prompt, retourne 400."""
        res = client.post('/api/engine/generate',
                          json={'device': 'cpu'},
                          content_type='application/json')
        assert res.status_code == 400

    def test_generate_active_apres_lancement(self, client):
        """La lampe passe au vert après lancement de la génération."""
        client.post('/api/engine/generate',
                    json={'prompt': 'Test lampe verte', 'device': 'cpu'},
                    content_type='application/json')
        # Petite pause pour laisser le thread démarrer
        time.sleep(0.05)
        res = client.get('/api/engine/status')
        data = res.get_json()
        assert data['active'] is True

    def test_generate_device_cpu_transmis(self, client):
        """Le device sélectionné est bien transmis au moteur."""
        client.post('/api/engine/generate',
                    json={'prompt': 'Test device CPU', 'device': 'cpu'},
                    content_type='application/json')
        time.sleep(0.05)
        res = client.get('/api/engine/status')
        data = res.get_json()
        assert data['device'] == 'cpu'

    def test_generate_inactive_apres_fin(self, client):
        """La lampe repasse au rouge après la fin de la génération (simulation ~2s)."""
        client.post('/api/engine/generate',
                    json={'prompt': 'Test fin génération', 'device': 'cpu'},
                    content_type='application/json')
        # La génération simule ~2s de traitement
        time.sleep(3.0)
        res = client.get('/api/engine/status')
        data = res.get_json()
        assert data['active'] is False


# ── Tests /api/engine/cancel ──────────────────────────────────────────────────

class TestEngineCancel:
    def test_cancel_reset_lampe(self, client):
        """Reset d'urgence remet la lampe au rouge."""
        # Forcer état actif manuellement
        with studio_app._engine_lock:
            studio_app._engine_state['is_generating'] = True
        res = client.post('/api/engine/cancel')
        assert res.status_code == 200
        data = res.get_json()
        assert data.get('ok') is True
        assert data.get('active') is False

    def test_cancel_idempotent(self, client):
        """Cancel fonctionne même si le moteur est déjà inactif."""
        res = client.post('/api/engine/cancel')
        assert res.status_code == 200


# ── Tests /api/health (existant + intégration) ────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        """L'endpoint health retourne status ok."""
        res = client.get('/api/health')
        assert res.status_code == 200
        data = res.get_json()
        assert data.get('status') == 'ok'

    def test_health_contient_version(self, client):
        """L'endpoint health retourne le numéro de version."""
        res = client.get('/api/health')
        data = res.get_json()
        assert 'version' in data


# ── Tests intégration complète ────────────────────────────────────────────────

class TestIntegration:
    def test_cycle_complet(self, client):
        """
        Cycle complet : sélection CPU → génération → lampe verte → fin → lampe rouge.
        """
        # 1. Sélection device CPU
        r = client.post('/api/engine/device', json={'device': 'cpu'},
                        content_type='application/json')
        assert r.status_code == 200

        # 2. Vérifier status initial
        r = client.get('/api/engine/status')
        assert r.get_json()['active'] is False

        # 3. Lancer génération
        r = client.post('/api/engine/generate',
                        json={'prompt': 'Cycle complet test', 'device': 'cpu'},
                        content_type='application/json')
        assert r.status_code == 202
        job_id = r.get_json()['job_id']
        assert job_id

        # 4. Lampe verte pendant génération
        time.sleep(0.1)
        r = client.get('/api/engine/status')
        assert r.get_json()['active'] is True

        # 5. Attendre fin
        time.sleep(3.0)
        r = client.get('/api/engine/status')
        assert r.get_json()['active'] is False

    def test_cancel_interrompt_generation(self, client):
        """Cancel interrompt visuellement la génération (reset lampe)."""
        client.post('/api/engine/generate',
                    json={'prompt': 'Test annulation', 'device': 'cpu'},
                    content_type='application/json')
        time.sleep(0.1)
        # Annuler
        client.post('/api/engine/cancel')
        r = client.get('/api/engine/status')
        assert r.get_json()['active'] is False
