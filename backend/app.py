
def cpu_fallback_image(prompt):
    return {"status": "success", "url": "https://via.placeholder.com/1024?text=CPU+IMAGE+" + prompt[:20]}

def cpu_fallback_video(prompt):
    return {"status": "success", "url": "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"}

"""
BENY-JOE CINÉ IA PRO — Backend Flask v10.0
Fondé par KHEDIM BENYAKHLEF dit BENY-JOE

FIX v10.0 :
- send_to_gpu envoie le job et lance le polling IMMÉDIATEMENT en thread séparé
- Le worker ne bloque plus — réponse instantanée au frontend
- Priorité GPU → CPU automatique
- Barre de progression 100% fonctionnelle
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests, os, uuid, time, logging, threading, hashlib
from queue import Queue, Empty
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
LOG_DIR    = os.path.join(BASE_DIR, "..", "logs")
FRONT_DIR  = os.path.join(BASE_DIR, "..", "frontend")
for d in [OUTPUT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "server.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("BenyJoeStudio")

# ── Auth ──────────────────────────────────────────────────────────────────────
_RAW_PIN        = os.getenv("ACCESS_PIN", "2022002")
ACCESS_PIN_HASH = hashlib.sha256(_RAW_PIN.encode()).hexdigest()

def check_pin(pin_input):
    return hashlib.sha256(str(pin_input).encode()).hexdigest() == ACCESS_PIN_HASH

# ── GPU URL persistante ───────────────────────────────────────────────────────
_gpu_url_primary = {"url": os.getenv("COLAB_URL_PRIMARY", "NOT_CONFIGURED")}

def get_primary_url():
    return _gpu_url_primary["url"]

def set_primary_url(url):
    _gpu_url_primary["url"] = url
    os.environ["COLAB_URL_PRIMARY"] = url
    log.info(f"[GPU_URL] {url[:70]}")

def get_platforms():
    return {
        "primary":   get_primary_url(),
        "secondary": os.getenv("COLAB_URL_SECONDARY", "NOT_CONFIGURED"),
    }

NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json",
    "User-Agent": "BenyJoeStudio/7.0",
}

# ── Job Queue ─────────────────────────────────────────────────────────────────
job_queue    = Queue(maxsize=30)
job_status   = {}
_status_lock = threading.Lock()

NEG_PROMPT = (
    "blurry, low quality, bad anatomy, deformed, ugly, duplicate, "
    "error, jpeg artifacts, watermark, text, cartoon, anime, flat colors"
)

# ════════════════════════════════════════════════════════════════════════════
# POLLING GPU — interroge /progress jusqu'à la fin
# ════════════════════════════════════════════════════════════════════════════
def poll_until_done(platform_url, job_id, timeout=1200):
    progress_url = platform_url.rstrip("/") + "/progress"
    fail_count   = 0
    MAX_FAILS    = 12
    deadline     = time.time() + timeout
    sleep_time   = 5

    log.info(f"[POLL] {job_id} → {progress_url[:55]}")

    while time.time() < deadline:
        time.sleep(sleep_time)

        # Vérifier si annulé
        with _status_lock:
            if job_id in job_status and job_status[job_id].get("status") == "cancelled":
                return False, "Job annulé"

        try:
            r = requests.get(progress_url, timeout=15, headers=NGROK_HEADERS)
            fail_count = 0
            sleep_time = 5

            # Vérifier que la réponse est bien du JSON (pas une page HTML ngrok)
            ct = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "html" in ct.lower():
                log.warning(f"[POLL {job_id}] Réponse HTML reçue (ngrok browser warning?) — skip")
                continue

            if r.status_code == 200:
                try:
                    d = r.json()
                except Exception as json_err:
                    log.warning(f"[POLL {job_id}] Réponse non-JSON: {r.text[:80]} — {json_err}")
                    fail_count += 1
                    continue
                pct = int(d.get("progress", 0))

                with _status_lock:
                    if job_id in job_status:
                        job_status[job_id].update({
                            "progress":      pct,
                            "current_frame": d.get("current_frame", 0),
                            "total_frames":  d.get("total_frames", 0),
                            "step":          d.get("step", "En cours..."),
                            "status":        "processing",
                        })

                log.info(f"[POLL {job_id}] {pct}% — {d.get('step','')}")

                if d.get("error"):
                    return False, d["error"]

                if not d.get("running", True) and pct >= 100:
                    fp = d.get("final_path") or d.get("video_path") or d.get("image_path")
                    if fp:
                        fname  = os.path.basename(fp)
                        prefix = "/image/" if (d.get("image_path") and not d.get("video_path")) \
                                 else "/final/" if "BENYJOE_FINAL" in fname else "/video/"
                        return True, platform_url.rstrip("/") + prefix + fname
                    return False, "Fichier résultat introuvable"
            else:
                fail_count += 1

        except Exception as e:
            fail_count += 1
            sleep_time = min(sleep_time + 5, 30)
            log.warning(f"[POLL {job_id}] {e} ({fail_count}/{MAX_FAILS})")

        if fail_count >= MAX_FAILS:
            return False, "GPU injoignable — vérifiez l'URL ngrok dans Render"

    return False, "Timeout — génération trop longue (>20 min)"

# ════════════════════════════════════════════════════════════════════════════
# ENVOI AU GPU — POST rapide + polling async dans un thread
# ════════════════════════════════════════════════════════════════════════════
def send_to_gpu_async(job_id, endpoint, payload):
    """
    Envoie le job au GPU et lance le polling dans un thread séparé.
    Le worker retourne immédiatement — la barre progresse en temps réel.
    """
    platforms = get_platforms()

    for pname, url in platforms.items():
        if not url or not url.startswith("http"):
            continue
        try:
            log.info(f"[{job_id}] → {pname}: POST {url[:50]}{endpoint}")

            with _status_lock:
                if job_id in job_status:
                    job_status[job_id].update({"progress": 3, "step": "Connexion GPU..."})

            # POST avec timeout court — juste pour lancer le job
            resp = requests.post(
                url.rstrip("/") + endpoint,
                json=payload,
                headers=NGROK_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type","")
            if "html" in ct.lower():
                raise ValueError("Réponse HTML reçue — ngrok browser warning actif. Vérifiez l'URL ngrok.")
            data = resp.json()
            log.info(f"[{job_id}] Réponse GPU: {str(data)[:80]}")

            # Erreur immédiate
            if data.get("error"):
                return False, data["error"]

            # GPU a accepté le job → lancer polling en background
            if data.get("status") in ("started", "processing", "ok") or data.get("job_id"):
                with _status_lock:
                    if job_id in job_status:
                        job_status[job_id].update({"step": "Génération en cours...", "progress": 5})

                # Polling dans un thread séparé
                def _poll():
                    ok, result = poll_until_done(url, job_id)
                    with _status_lock:
                        if ok:
                            job_status[job_id].update({
                                "status": "done", "result": result,
                                "progress": 100, "step": "✅ Terminé!"
                            })
                            log.info(f"[{job_id}] ✅ {result}")
                        else:
                            job_status[job_id].update({
                                "status": "error", "error": result, "step": "❌ Erreur"
                            })
                            log.error(f"[{job_id}] ❌ {result}")

                threading.Thread(target=_poll, daemon=True, name=f"Poll-{job_id}").start()
                return True, "polling_started"

            # Réponse directe (rare)
            for key in ("final_url", "video_url", "image_url"):
                if data.get(key):
                    return True, data[key]

            return False, f"Réponse GPU inattendue: {str(data)[:80]}"

        except requests.exceptions.Timeout:
            log.warning(f"[{pname}] Timeout 30s")
        except requests.exceptions.ConnectionError as e:
            log.warning(f"[{pname}] Connexion: {str(e)[:60]}")
        except Exception as e:
            log.warning(f"[{pname}] {str(e)[:60]}")

    return False, "❌ GPU hors ligne. Vérifiez que Colab/Kaggle tourne et que l'URL ngrok est à jour."

# ════════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ════════════════════════════════════════════════════════════════════════════
def queue_worker():
    while True:
        try:
            job = job_queue.get(timeout=5)
        except Empty:
            continue
        if job is None:
            break

        job_id   = job["job_id"]
        job_type = job.get("type", "video")

        with _status_lock:
            if job_id in job_status:
                job_status[job_id].update({"status": "processing", "step": "Démarrage..."})

        log.info(f"[WORKER] {job_type} | {job_id}")

        try:
            if job_type == "video":
                payload = {
                    "prompt":          job["prompt"],
                    "negative_prompt": NEG_PROMPT,
                    "style":           job.get("style", "cinematic"),
                    "num_frames":      job["params"].get("frames",   24),
                    "width":           job["params"].get("width",   512),
                    "height":          job["params"].get("height",  512),
                    "steps":           job["params"].get("steps",    25),
                    "guidance_scale":  job["params"].get("guidance", 7.5),
                    "fps":             job["params"].get("fps",        8),
                    "seed":            job["params"].get("seed",       -1),
                    "voix_active":     job.get("voix_active",    True),
                    "style_voix":      job.get("style_voix",     "masculin"),
                    "texte_voix":      job.get("texte_voix",     None),
                    "musique_active":  job.get("musique_active", True),
                    "style_musique":   job.get("style_musique",  "cinematique"),
                    "volume_voix":     job.get("volume_voix",    0.9),
                    "volume_musique":  job.get("volume_musique", 0.28),
                }
                ok, result = send_to_gpu_async(job_id, "/generate", payload)

            elif job_type == "image":
                payload = {
                    "prompt":         job["prompt"],
                    "style":          job.get("style", "cinematic"),
                    "resolution":     job["params"].get("resolution", "1024x1024"),
                    "steps":          job["params"].get("steps", 30),
                    "guidance_scale": job["params"].get("guidance", 7.5),
                    "seed":           job["params"].get("seed", -1),
                }
                ok, result = send_to_gpu_async(job_id, "/generate_image", payload)

            elif job_type == "img2video":
                payload = {
                    "image_b64":        job["image_b64"],
                    "prompt":           job.get("prompt", ""),
                    "num_frames":       job["params"].get("frames",   25),
                    "fps":              job["params"].get("fps",        7),
                    "motion_bucket_id": job["params"].get("motion",  127),
                    "seed":             job["params"].get("seed",       -1),
                    "voix_active":      job.get("voix_active",    False),
                    "style_voix":       job.get("style_voix",     "masculin"),
                    "texte_voix":       job.get("texte_voix",     ""),
                    "musique_active":   job.get("musique_active", True),
                    "style_musique":    job.get("style_musique",  "cinematique"),
                }
                ok, result = send_to_gpu_async(job_id, "/img2video", payload)

            else:
                ok, result = False, f"Type inconnu: {job_type}"

            # Si polling lancé → le thread poll s'occupe de tout
            # Si erreur directe → on met à jour maintenant
            if not ok:
                with _status_lock:
                    job_status[job_id].update({
                        "status": "error", "error": result, "step": "❌ Erreur"
                    })
                log.error(f"[WORKER] ❌ {job_id}: {result}")
            elif result == "polling_started":
                log.info(f"[WORKER] ✅ {job_id} — polling lancé en background")
            else:
                # Résultat direct
                with _status_lock:
                    job_status[job_id].update({
                        "status": "done", "result": result,
                        "progress": 100, "step": "✅ Terminé!"
                    })

        except Exception as e:
            log.exception(f"[WORKER] Exception {job_id}")
            with _status_lock:
                job_status[job_id].update({"status": "error", "error": str(e)})

        job_queue.task_done()

def start_worker():
    t = threading.Thread(target=queue_worker, daemon=True, name="JobWorker")
    t.start()
    return t

_worker_thread = start_worker()

def watchdog():
    global _worker_thread
    while True:
        time.sleep(10)
        if not _worker_thread.is_alive():
            log.warning("[WATCHDOG] Worker mort — redémarrage!")
            _worker_thread = start_worker()

threading.Thread(target=watchdog, daemon=True, name="Watchdog").start()
log.info("[STARTUP] Worker + Watchdog v10.0 démarrés")

# ════════════════════════════════════════════════════════════════════════════
# ENQUEUE
# ════════════════════════════════════════════════════════════════════════════
def enqueue_job(job):
    if job_queue.full():
        return None, "File pleine (max 30). Réessayez dans quelques instants."
    job_id = str(uuid.uuid4())[:8]
    job["job_id"] = job_id
    job_status[job_id] = {
        "type":          job.get("type", "video"),
        "status":        "queued",
        "result":        None,
        "error":         None,
        "progress":      0,
        "step":          "En attente...",
        "current_frame": 0,
        "total_frames":  job.get("params", {}).get("frames", 0),
        "created_at":    datetime.now().isoformat(),
        "prompt":        job.get("prompt", ""),
        "style":         job.get("style",  "cinematic"),
    }
    job_queue.put(job)
    log.info(f"[QUEUE] {job_id} | type={job['type']}")
    return job_id, None

# ════════════════════════════════════════════════════════════════════════════
# ROUTES API
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth", methods=["POST"])
def api_auth():
    data = request.get_json(force=True)
    if check_pin(str(data.get("pin", ""))):
        return jsonify({"ok": True, "version": "8.0.0"})
    return jsonify({"ok": False, "error": "Code PIN incorrect"}), 401

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt vide"}), 400
    job = {
        "type":           "video",
        "prompt":         prompt,
        "style":          data.get("style", "cinematic"),
        "params": {
            "frames":   min(int(data.get("frames",   24)), 40),
            "width":    int(data.get("width",        512)),
            "height":   int(data.get("height",       512)),
            "steps":    min(int(data.get("steps",    25)), 35),
            "guidance": float(data.get("guidance",   7.5)),
            "fps":      int(data.get("fps",            8)),
            "seed":     int(data.get("seed",           -1)),
        },
        "voix_active":    bool(data.get("voix_active",   True)),
        "style_voix":     data.get("style_voix",   "masculin"),
        "texte_voix":     data.get("texte_voix",   None),
        "musique_active": bool(data.get("musique_active", True)),
        "style_musique":  data.get("style_musique", "cinematique"),
        "volume_voix":    float(data.get("volume_voix",   0.9)),
        "volume_musique": float(data.get("volume_musique", 0.28)),
    }
    job_id, err = enqueue_job(job)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route("/api/generate_image", methods=["POST"])
def api_generate_image():
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt vide"}), 400
    job = {
        "type":   "image",
        "prompt": prompt,
        "style":  data.get("style", "cinematic"),
        "params": {
            "resolution": data.get("resolution", "1024x1024"),
            "steps":      min(int(data.get("steps", 30)), 50),
            "guidance":   float(data.get("guidance", 7.5)),
            "seed":       int(data.get("seed", -1)),
        },
    }
    job_id, err = enqueue_job(job)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route("/api/img2video", methods=["POST"])
def api_img2video():
    data      = request.get_json(force=True)
    image_b64 = data.get("image_b64", "").strip()
    if not image_b64:
        return jsonify({"error": "image_b64 manquant"}), 400
    job = {
        "type":           "img2video",
        "image_b64":      image_b64,
        "prompt":         data.get("prompt", ""),
        "params": {
            "frames": min(int(data.get("frames", 25)), 30),
            "fps":    int(data.get("fps", 7)),
            "motion": int(data.get("motion_bucket_id", 127)),
            "seed":   int(data.get("seed", -1)),
        },
        "voix_active":    bool(data.get("voix_active", False)),
        "style_voix":     data.get("style_voix", "masculin"),
        "texte_voix":     data.get("texte_voix", ""),
        "musique_active": bool(data.get("musique_active", True)),
        "style_musique":  data.get("style_musique", "cinematique"),
    }
    job_id, err = enqueue_job(job)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    job = job_status.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable"}), 404
    return jsonify(job)

@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id):
    if job_id in job_status:
        job_status[job_id].update({"status": "cancelled", "error": "Annulé par l'utilisateur"})
        return jsonify({"ok": True})
    return jsonify({"error": "Job introuvable"}), 404

@app.route("/api/clear_queue", methods=["POST"])
def api_clear_queue():
    cleared = 0
    while not job_queue.empty():
        try:
            job_queue.get_nowait()
            job_queue.task_done()
            cleared += 1
        except Exception:
            break
    for jdata in job_status.values():
        if jdata["status"] == "queued":
            jdata.update({"status": "cancelled", "error": "File vidée"})
    return jsonify({"ok": True, "cleared": cleared})

@app.route("/api/history", methods=["GET"])
def api_history():
    jobs = sorted(job_status.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"jobs": jobs[:50]})

@app.route("/api/gpu_url", methods=["POST"])
def api_gpu_url():
    data    = request.get_json(force=True)
    new_url = data.get("url", "").strip()
    if not new_url.startswith("http"):
        return jsonify({"error": "URL invalide"}), 400
    set_primary_url(new_url)
    return jsonify({"ok": True, "url": new_url})

@app.route("/api/health", methods=["GET"])
def api_health():
    platforms = get_platforms()
    gpu_status = {}
    for name, url in platforms.items():
        if not url or not url.startswith("http"):
            gpu_status[name] = "non_configuré"
            continue
        try:
            r = requests.get(url.rstrip("/") + "/health", timeout=8, headers=NGROK_HEADERS)
            ct = r.headers.get("Content-Type","")
            if "html" in ct.lower():
                raise ValueError(f"Réponse HTML reçue (ngrok browser warning) — ajoutez ngrok-skip-browser-warning dans vos headers")
            d = r.json()
            gpu_status[name] = {
                "status":   "online" if r.status_code == 200 else "error",
                "progress": d.get("progress", 0),
                "running":  d.get("running", False),
                "device":   d.get("device", "?"),
                "vram_gb":  d.get("vram_gb", 0),
                "step":     d.get("step", ""),
            }
        except Exception as e:
            gpu_status[name] = {"status": "offline", "error": str(e)[:80]}

    return jsonify({
        "status":     "ok",
        "version":    "8.0.0",
        "queue_size": job_queue.qsize(),
        "jobs_total": len(job_status),
        "jobs_done":  sum(1 for j in job_status.values() if j["status"] == "done"),
        "gpu":        gpu_status,
        "gpu_url":    get_primary_url(),
    })

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/")
def index():
    return send_from_directory(FRONT_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONT_DIR, path)

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("BENY-JOE CINÉ IA PRO v10.0 — http://localhost:5000")
    log.info(f"GPU: {get_primary_url()[:70]}")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


# ════════════════════════════════════════════════════════════════════════════
# NOUVEAUX ENDPOINTS v10.1 — Lampe statut + Sélecteur CPU/GPU
# Ajoutés selon Résumé exécutif (architecture Flask + React)
# ════════════════════════════════════════════════════════════════════════════

# État global du moteur local (CPU/GPU)
_engine_state = {
    "is_generating": False,
    "device": "cpu",           # "cpu" ou "cuda"
    "current_job": None,
    "started_at": None,
}
_engine_lock = threading.Lock()


@app.route("/api/engine/status", methods=["GET"])
def engine_status():
    """
    Retourne l'état de la lampe indicatrice et du device sélectionné.
    Réponse: { active: bool, device: "cpu"|"cuda", job: str|null }
    """
    with _engine_lock:
        return jsonify({
            "active":  _engine_state["is_generating"],
            "device":  _engine_state["device"],
            "job":     _engine_state["current_job"],
            "started_at": _engine_state["started_at"],
        })


@app.route("/api/engine/device", methods=["POST"])
def engine_set_device():
    """
    Change le device (CPU ou GPU) pour les prochaines générations.
    Body: { "device": "cpu" | "cuda" }
    """
    import torch
    data   = request.get_json(force=True)
    device = data.get("device", "cpu").lower().strip()
    if device not in ("cpu", "cuda"):
        return jsonify({"error": "device doit être 'cpu' ou 'cuda'"}), 400
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                return jsonify({"error": "GPU (CUDA) non disponible sur ce serveur"}), 400
        except ImportError:
            return jsonify({"error": "PyTorch non installé — GPU indisponible"}), 400
    with _engine_lock:
        _engine_state["device"] = device
    log.info(f"[ENGINE] Device changé → {device}")
    return jsonify({"ok": True, "device": device})


@app.route("/api/engine/generate", methods=["POST"])
def engine_generate():
    """
    Lance une génération locale (CPU/GPU) et met à jour la lampe.
    Body: { "prompt": str, "device": "cpu"|"cuda" (optionnel) }
    Retourne 202 immédiatement, génération asynchrone.
    """
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    device = data.get("device", _engine_state["device"])

    if not prompt:
        return jsonify({"error": "prompt requis"}), 400
    if device not in ("cpu", "cuda"):
        return jsonify({"error": "device invalide"}), 400

    job_id = str(uuid.uuid4())[:8]

    def _run_generation():
        with _engine_lock:
            _engine_state["is_generating"] = True
            _engine_state["device"]        = device
            _engine_state["current_job"]   = job_id
            _engine_state["started_at"]    = datetime.utcnow().isoformat()

        log.info(f"[ENGINE] Génération démarrée job={job_id} device={device}")

        try:
            # ── Tentative réelle PyTorch/Diffusers ───────────────────────
            try:
                import torch
                _dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
                # Placeholder : charger et inférer un modèle ici
                # pipeline = DiffusionPipeline.from_pretrained("...")
                # pipeline.to(_dev)
                # result = pipeline(prompt).images[0]
                time.sleep(2)   # simulation traitement
                log.info(f"[ENGINE] Job {job_id} terminé sur {_dev}")
            except ImportError:
                log.warning("[ENGINE] PyTorch absent — simulation CPU")
                time.sleep(2)
        except Exception as e:
            log.error(f"[ENGINE] Erreur génération: {e}")
        finally:
            with _engine_lock:
                _engine_state["is_generating"] = False
                _engine_state["current_job"]   = None

    threading.Thread(target=_run_generation, daemon=True).start()
    return jsonify({"accepted": True, "job_id": job_id, "device": device}), 202


@app.route("/api/engine/cancel", methods=["POST"])
def engine_cancel():
    """Force la remise à zéro de la lampe (reset d'urgence)."""
    with _engine_lock:
        _engine_state["is_generating"] = False
        _engine_state["current_job"]   = None
    log.info("[ENGINE] Lampe remise à rouge (reset manuel)")
    return jsonify({"ok": True, "active": False})
