"""
BENY-JOE CINÉ IA PRO v5.0 — Suite de tests automatisés
Testez votre installation avant de déployer
Usage : python tests/test_studio.py --url http://localhost:5000 --pin 2022002
"""

import argparse, requests, time, json, base64, os, sys
from datetime import datetime

GREEN = "\033[92m"
RED   = "\033[91m"
GOLD  = "\033[93m"
CYAN  = "\033[96m"
RESET = "\033[0m"

def ok(msg):   print(f"{GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"{RED}❌ {msg}{RESET}")
def info(msg): print(f"{CYAN}ℹ️  {msg}{RESET}")
def warn(msg): print(f"{GOLD}⚠️  {msg}{RESET}")


class StudioTester:
    def __init__(self, base_url, pin):
        self.base = base_url.rstrip("/")
        self.pin  = pin
        self.results = []

    def check(self, name, fn):
        try:
            fn()
            ok(name)
            self.results.append((name, True, None))
        except AssertionError as e:
            fail(f"{name}: {e}")
            self.results.append((name, False, str(e)))
        except Exception as e:
            fail(f"{name}: {e}")
            self.results.append((name, False, str(e)))

    def test_health(self):
        def _():
            r = requests.get(self.base + "/api/health", timeout=10)
            assert r.status_code == 200, f"HTTP {r.status_code}"
            d = r.json()
            assert d["status"] == "ok", f"status={d['status']}"
            assert "version" in d, "version manquante"
            assert "gpu" in d, "gpu manquant"
            info(f"Version: {d['version']} | Queue: {d.get('queue_size',0)} | Jobs: {d.get('jobs_total',0)}")
        self.check("Health endpoint", _)

    def test_auth_ok(self):
        def _():
            r = requests.post(self.base + "/api/auth",
                              json={"pin": self.pin}, timeout=10)
            assert r.status_code == 200, f"HTTP {r.status_code}"
            assert r.json()["ok"], f"Auth refusée: {r.json()}"
        self.check("Auth PIN correct", _)

    def test_auth_fail(self):
        def _():
            r = requests.post(self.base + "/api/auth",
                              json={"pin": "000000"}, timeout=10)
            assert r.status_code == 401, f"Devrait être 401, got {r.status_code}"
            assert not r.json()["ok"], "Devrait refuser un mauvais PIN"
        self.check("Auth PIN incorrect (doit échouer)", _)

    def test_generate_video_queued(self):
        def _():
            r = requests.post(self.base + "/api/generate",
                              json={
                                  "prompt": "Test cinematic scene",
                                  "style":  "cinematic",
                                  "frames": 8,
                                  "steps":  10,
                                  "voix_active":    False,
                                  "musique_active": False,
                              }, timeout=15)
            assert r.status_code == 200, f"HTTP {r.status_code} — {r.text}"
            d = r.json()
            assert "job_id" in d, f"job_id manquant: {d}"
            assert d["status"] == "queued", f"status={d['status']}"
            info(f"Job vidéo créé: {d['job_id']}")
            # Annuler immédiatement pour ne pas saturer
            requests.post(self.base + f"/api/cancel/{d['job_id']}", timeout=5)
        self.check("Génération vidéo — création job", _)

    def test_generate_image_queued(self):
        def _():
            r = requests.post(self.base + "/api/generate_image",
                              json={
                                  "prompt": "Test image portrait",
                                  "style":  "cinematic",
                                  "resolution": "768x768",
                                  "steps": 5,
                              }, timeout=15)
            assert r.status_code == 200, f"HTTP {r.status_code}"
            d = r.json()
            assert "job_id" in d, f"job_id manquant: {d}"
            info(f"Job image créé: {d['job_id']}")
            requests.post(self.base + f"/api/cancel/{d['job_id']}", timeout=5)
        self.check("Génération image — création job", _)

    def test_img2video_no_image(self):
        def _():
            r = requests.post(self.base + "/api/img2video",
                              json={"prompt": "test"}, timeout=10)
            assert r.status_code == 400, f"Devrait être 400, got {r.status_code}"
        self.check("Img2Video sans image (doit refuser)", _)

    def test_img2video_with_image(self):
        def _():
            # Image 1x1 pixel blanc en base64
            img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            r = requests.post(self.base + "/api/img2video",
                              json={"image_b64": img_b64, "prompt": "test", "num_frames": 5},
                              timeout=15)
            assert r.status_code == 200, f"HTTP {r.status_code}"
            d = r.json()
            assert "job_id" in d, f"job_id manquant: {d}"
            info(f"Job img2video créé: {d['job_id']}")
            requests.post(self.base + f"/api/cancel/{d['job_id']}", timeout=5)
        self.check("Img2Video avec image — création job", _)

    def test_status_not_found(self):
        def _():
            r = requests.get(self.base + "/api/status/INEXISTANT", timeout=10)
            assert r.status_code == 404, f"Devrait être 404, got {r.status_code}"
        self.check("Status job inexistant (doit 404)", _)

    def test_clear_queue(self):
        def _():
            r = requests.post(self.base + "/api/clear_queue", timeout=10)
            assert r.status_code == 200, f"HTTP {r.status_code}"
            assert r.json()["ok"], "clear_queue échoué"
        self.check("Clear queue", _)

    def test_history(self):
        def _():
            r = requests.get(self.base + "/api/history", timeout=10)
            assert r.status_code == 200, f"HTTP {r.status_code}"
            assert "jobs" in r.json(), "jobs manquant"
        self.check("Historique jobs", _)

    def test_gpu_url_invalid(self):
        def _():
            r = requests.post(self.base + "/api/gpu_url",
                              json={"url": "pas_une_url"}, timeout=10)
            assert r.status_code == 400, f"Devrait refuser une URL invalide"
        self.check("GPU URL invalide (doit refuser)", _)

    def test_frontend(self):
        def _():
            r = requests.get(self.base + "/", timeout=10)
            assert r.status_code == 200, f"HTTP {r.status_code}"
            assert "BENY-JOE" in r.text, "Frontend ne contient pas BENY-JOE"
        self.check("Frontend chargé", _)

    def test_prompt_empty(self):
        def _():
            r = requests.post(self.base + "/api/generate",
                              json={"prompt": ""}, timeout=10)
            assert r.status_code == 400, f"Devrait refuser un prompt vide"
        self.check("Prompt vide (doit refuser)", _)

    def run_all(self):
        print(f"\n{'='*55}")
        print(f"  BENY-JOE CINÉ IA PRO v5.0 — Tests automatisés")
        print(f"  URL: {self.base}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*55}\n")

        tests = [
            self.test_health,
            self.test_auth_ok,
            self.test_auth_fail,
            self.test_frontend,
            self.test_prompt_empty,
            self.test_generate_video_queued,
            self.test_generate_image_queued,
            self.test_img2video_no_image,
            self.test_img2video_with_image,
            self.test_status_not_found,
            self.test_clear_queue,
            self.test_history,
            self.test_gpu_url_invalid,
        ]

        for test in tests:
            test()

        passed = sum(1 for _, ok, _ in self.results if ok)
        total  = len(self.results)
        pct    = round(passed / total * 100) if total else 0

        print(f"\n{'='*55}")
        print(f"  Résultats: {passed}/{total} tests réussis ({pct}%)")
        if passed == total:
            print(f"  {GREEN}🎉 Tous les tests passent!{RESET}")
        else:
            print(f"  {RED}⚠️  {total-passed} tests échoués:{RESET}")
            for name, ok_, err in self.results:
                if not ok_:
                    print(f"     • {name}: {err}")
        print(f"{'='*55}\n")
        return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests BENY-JOE Studio v5.0")
    parser.add_argument("--url", default="http://localhost:5000", help="URL du serveur")
    parser.add_argument("--pin", default="2022002", help="Code PIN")
    args = parser.parse_args()

    tester = StudioTester(args.url, args.pin)
    success = tester.run_all()
    sys.exit(0 if success else 1)
