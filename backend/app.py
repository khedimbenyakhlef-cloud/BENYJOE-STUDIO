"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BENY-JOE CINÉ IA PRO — Backend Flask v5.0                           ║
║         Fondé par KHEDIM BENYAKHLEF dit BENY-JOE                            ║
║                                                                              ║
║  Architecture :                                                              ║
║  • File de jobs async (Queue + Worker thread)                                ║
║  • Polling Kaggle GPU intelligent (retry + timeout adaptatif)                ║
║  • Auth PIN hashé SHA-256                                                    ║
║  • Rechargement dynamique URL ngrok (sans redémarrer Render)                 ║
║  • Routes : /api/generate, /api/generate_image, /api/img2video               ║
║             /api/status, /api/health, /api/gpu_url, /api/cancel              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests, os, uuid, time, logging, threading, hashlib, json
from queue import Queue, Empty
from datetime import datetime

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
LOG_DIR    = os.path.join(BASE_DIR, "..", "logs")
FRONT_DIR  = os.path.join(BASE_DIR, "..", "frontend")
for d in [OUTPUT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "server.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("BenyJoeStudio")

# ── Auth PIN ──────────────────────────────────────────────────────────────────
_RAW_PIN        = os.getenv("ACCESS_PIN", "2022002")
ACCESS_PIN_HASH = hashlib.sha256(_RAW_PIN.encode()).hexdigest()

def check_pin(pin_input: str) -> bool:
    return hashlib.sha256(str(pin_input).encode()).hexdigest() == ACCESS_PIN_HASH

# ── GPU Platforms (rechargement dynamique) ────────────────────────────────────
def get_platforms() -> dict:
    return {
        "primary":   os.getenv("COLAB_URL_PRIMARY",   "PUT_YOUR_NGROK_URL_HERE"),
        "secondary": os.getenv("COLAB_URL_SECONDARY", "NOT_CONFIGURED"),
    }

NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json",
    "User-Agent": "BenyJoeStudio/5.0",
}

# ── Job Queue ─────────────────────────────────────────────────────────────────
job_queue  : Queue = Queue(maxsize=30)
job_status : dict  = {}
_status_lock = threading.Lock()

# ── Style suffixes ────────────────────────────────────────────────────────────
STYLE_SUFFIXES = {
    "cinematic":   "cinematic lighting, dramatic shadows, anamorphic lens, film grain, 4K HDR",
    "cyberpunk":   "neon lights, rain-soaked streets, volumetric fog, ultra realistic, 8K",
    "epic_battle": "epic battle, slow motion, fire and smoke, dramatic sky, hyperrealistic, 8K",
    "nature":      "golden hour, misty mountains, ultra wide angle, cinematic grade, 4K",
    "scifi":       "space station, zero gravity, distant nebula, sci-fi tech, cinematic",
    "noir":        "black and white film noir, rain, smoky atmosphere, dramatic shadows, 1940s",
    "fantasy":     "enchanted forest, magical creatures, bioluminescent, epic fantasy, 8K",
    "horror":      "abandoned building, fog, moonlight, horror atmosphere, dark, ultra realistic",
    "romantique":  "golden hour, warm bokeh, emotional, dreamy, soft light, romantic, 4K",
}

NEG_PROMPT = (
    "blurry, low quality, bad anatomy, deformed, ugly, duplicate, "
    "error, jpeg artifacts, watermark, text, cartoon, anime, flat colors"
)


# ════════════════════════════════════════════════════════════════════════════
# POLLING GPU
# ════════════════════════════════════════════════════════════════════════════
def poll_until_done(platform_url: str, job_id: str, timeout: int = 720):
    """Interroge /progress sur le GPU jusqu'à la fin ou l'erreur."""
    progress_url = platform_url.rstrip("/") + "/progress"
    fail_count   = 0
    MAX_FAILS    = 10
    deadline     = time.time() + timeout

    while time.time() < deadline:
        time.sleep(7)
        try:
            r = requests.get(progress_url, timeout=14, headers=NGROK_HEADERS)
            fail_count = 0
            if r.status_code == 200:
                d = r.json()
                pct = d.get("progress", 0)
                with _status_lock:
                    if job_id in job_status:
                        job_status[job_id]["progress"]      = pct
                        job_status[job_id]["current_frame"] = d.get("current_frame", 0)
                        job_status[job_id]["total_frames"]  = d.get("total_frames",  0)
                        job_status[job_id]["step"]          = d.get("step", "")
                log.info(f"[{job_id}] {pct}% — {d.get('step', '')}")

                if d.get("error"):
                    return False, d["error"]

                if not d.get("running", True) and pct >= 100:
                    final_path = d.get("final_path") or d.get("video_path") or d.get("image_path")
                    if final_path:
                        fname = os.path.basename(final_path)
                        prefix = "/final/" if "BENYJOE_FINAL" in fname else "/video/"
                        if d.get("image_path") and not d.get("video_path"):
                            prefix = "/image/"
                        return True, platform_url.rstrip("/") + prefix + fname
                    return False, "Fichier résultat introuvable sur le GPU"

        except Exception as e:
            fail_count += 1
            log.warning(f"[{job_id}] Poll fail {fail_count}/{MAX_FAILS}: {e}")
            if fail_count >= MAX_FAILS:
                return False, f"GPU injoignable ({MAX_FAILS} tentatives): {e}"

    return False, "Timeout — génération trop longue (>12 min)"


# ════════════════════════════════════════════════════════════════════════════
# ENVOI D'UN JOB AU GPU
# ════════════════════════════════════════════════════════════════════════════
def send_to_gpu(job_id: str, endpoint: str, payload: dict, timeout: int = 360):
    """Envoie un POST au GPU Kaggle, gère réponse directe ET polling async."""
    platforms = get_platforms()

    log.info(f"[SEND_GPU] job={job_id} endpoint={endpoint} platforms={list(platforms.items())}")
    for pname, url in platforms.items():
        if not url.startswith("http"):
            log.warning(f"[SEND_GPU] {pname} URL invalide: {url[:50]}")
            continue
        try:
            log.info(f"[{job_id}] → {pname}: POST {endpoint} url={url[:60]}")
            with _status_lock:
                if job_id in job_status:
                    job_status[job_id]["progress"] = 3

            resp = requests.post(
                url.rstrip("/") + endpoint,
                json=payload,
                headers=NGROK_HEADERS,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Réponse directe (synchrone)
            for key in ("final_url", "video_url", "image_url"):
                if data.get(key):
                    log.info(f"[{job_id}] Réponse directe: {data[key]}")
                    return {"success": True, "result_url": data[key]}

            # Réponse asynchrone → polling
            if data.get("status") == "started" or data.get("job_id"):
                log.info(f"[{job_id}] Mode async → polling...")
                ok, result = poll_until_done(url, job_id)
                if ok:
                    return {"success": True, "result_url": result}
                return {"success": False, "error": result}

        except requests.exceptions.Timeout:
            log.warning(f"[{pname}] Timeout {timeout}s → backup")
        except requests.exceptions.ConnectionError as e:
            log.warning(f"[{pname}] Connexion: {e}")
        except Exception as e:
            log.warning(f"[{pname}] Erreur: {e}")

    return {
        "success": False,
        "error": (
            "❌ GPU hors ligne. Vérifiez que Kaggle tourne "
            "et que l'URL ngrok est à jour dans Render → Environment."
        ),
    }


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
                job_status[job_id]["status"] = "processing"

        log.info(f"[WORKER] {job_type} | {job_id} | {job.get('prompt','')[:60]}")

        try:
            if job_type == "video":
                payload  = {
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
                result = send_to_gpu(job_id, "/generate", payload, timeout=600)

            elif job_type == "image":
                payload = {
                    "prompt":          job["prompt"],
                    "style":           job.get("style", "cinematic"),
                    "resolution":      job["params"].get("resolution", "1024x1024"),
                    "steps":           job["params"].get("steps", 30),
                    "guidance_scale":  job["params"].get("guidance", 7.5),
                    "seed":            job["params"].get("seed", -1),
                }
                result = send_to_gpu(job_id, "/generate_image", payload, timeout=240)

            elif job_type == "img2video":
                payload = {
                    "image_b64":       job["image_b64"],
                    "prompt":          job.get("prompt", ""),
                    "num_frames":      job["params"].get("frames",   25),
                    "fps":             job["params"].get("fps",        7),
                    "motion_bucket_id": job["params"].get("motion",  127),
                    "seed":            job["params"].get("seed",       -1),
                    "voix_active":     job.get("voix_active",    False),
                    "style_voix":      job.get("style_voix",     "masculin"),
                    "texte_voix":      job.get("texte_voix",     ""),
                    "musique_active":  job.get("musique_active", True),
                    "style_musique":   job.get("style_musique",  "cinematique"),
                }
                result = send_to_gpu(job_id, "/img2video", payload, timeout=480)

            else:
                result = {"success": False, "error": f"Type inconnu: {job_type}"}

        except Exception as e:
            result = {"success": False, "error": str(e)}

        with _status_lock:
            if result["success"]:
                job_status[job_id].update({
                    "status":   "done",
                    "result":   result["result_url"],
                    "progress": 100,
                })
                log.info(f"[WORKER] ✅ {job_id} → {result['result_url']}")
            else:
                job_status[job_id].update({
                    "status": "error",
                    "error":  result["error"],
                })
                log.error(f"[WORKER] ❌ {job_id}: {result['error']}")

        job_queue.task_done()


# Démarrage du worker avec surveillance automatique
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
            log.warning("[WATCHDOG] Worker mort — redemarrage!")
            _worker_thread = start_worker()

threading.Thread(target=watchdog, daemon=True, name="Watchdog").start()
log.info("[STARTUP] Worker + Watchdog demarres")


# ════════════════════════════════════════════════════════════════════════════
# HELPER: créer un job et l'enqueue
# ════════════════════════════════════════════════════════════════════════════
def enqueue_job(job: dict) -> tuple:
    """Ajoute un job à la file. Retourne (job_id, error_msg)."""
    if job_queue.full():
        return None, "File pleine (max 30). Réessayez dans quelques instants."
    job_id = str(uuid.uuid4())[:8]
    job["job_id"] = job_id
    job_status[job_id] = {
        "type":       job.get("type", "video"),
        "status":     "queued",
        "result":     None,
        "error":      None,
        "progress":   0,
        "step":       "En attente...",
        "current_frame": 0,
        "total_frames":  job.get("params", {}).get("frames", 0),
        "created_at": datetime.now().isoformat(),
        "prompt":     job.get("prompt", ""),
        "style":      job.get("style",  "cinematic"),
    }
    job_queue.put(job)
    log.info(f"[QUEUE] {job_id} | type={job['type']} | queue={job_queue.qsize()}")
    return job_id, None


# ════════════════════════════════════════════════════════════════════════════
# ROUTES API
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth", methods=["POST"])
def api_auth():
    data = request.get_json(force=True)
    if check_pin(str(data.get("pin", ""))):
        return jsonify({"ok": True, "version": "5.0.0"})
    return jsonify({"ok": False, "error": "Code PIN incorrect"}), 401


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Génère: Texte → Vidéo + Voix + Musique"""
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt vide"}), 400

    params = {
        "frames":   min(int(data.get("frames",   24)), 40),
        "width":    int(data.get("width",        512)),
        "height":   int(data.get("height",       512)),
        "steps":    min(int(data.get("steps",    25)), 35),
        "guidance": float(data.get("guidance",   7.5)),
        "fps":      int(data.get("fps",            8)),
        "seed":     int(data.get("seed",           -1)),
    }
    job = {
        "type":          "video",
        "prompt":        prompt,
        "style":         data.get("style", "cinematic"),
        "params":        params,
        "voix_active":   bool(data.get("voix_active",   True)),
        "style_voix":    data.get("style_voix",   "masculin"),
        "texte_voix":    data.get("texte_voix",   None),
        "musique_active": bool(data.get("musique_active", True)),
        "style_musique": data.get("style_musique", "cinematique"),
        "volume_voix":   float(data.get("volume_voix",   0.9)),
        "volume_musique": float(data.get("volume_musique", 0.28)),
    }
    job_id, err = enqueue_job(job)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/generate_image", methods=["POST"])
def api_generate_image():
    """Génère: Texte → Image SDXL"""
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt vide"}), 400

    params = {
        "resolution": data.get("resolution", "1024x1024"),
        "steps":      min(int(data.get("steps", 30)), 50),
        "guidance":   float(data.get("guidance", 7.5)),
        "seed":       int(data.get("seed", -1)),
    }
    job = {
        "type":   "image",
        "prompt": prompt,
        "style":  data.get("style", "cinematic"),
        "params": params,
    }
    job_id, err = enqueue_job(job)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/img2video", methods=["POST"])
def api_img2video():
    """Anime une image → Vidéo (SVD)"""
    data      = request.get_json(force=True)
    image_b64 = data.get("image_b64", "").strip()
    if not image_b64:
        return jsonify({"error": "image_b64 manquant"}), 400

    params = {
        "frames": min(int(data.get("frames", 25)), 30),
        "fps":    int(data.get("fps", 7)),
        "motion": int(data.get("motion_bucket_id", 127)),
        "seed":   int(data.get("seed", -1)),
    }
    job = {
        "type":          "img2video",
        "image_b64":     image_b64,
        "prompt":        data.get("prompt", ""),
        "params":        params,
        "voix_active":   bool(data.get("voix_active", False)),
        "style_voix":    data.get("style_voix", "masculin"),
        "texte_voix":    data.get("texte_voix", ""),
        "musique_active": bool(data.get("musique_active", True)),
        "style_musique": data.get("style_musique", "cinematique"),
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
    for jid, jdata in job_status.items():
        if jdata["status"] == "queued":
            jdata.update({"status": "cancelled", "error": "File vidée"})
    log.info(f"[CLEAR] {cleared} jobs supprimés")
    return jsonify({"ok": True, "cleared": cleared})


@app.route("/api/history", methods=["GET"])
def api_history():
    """Retourne l'historique de tous les jobs"""
    jobs = sorted(job_status.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"jobs": jobs[:50]})


@app.route("/api/gpu_url", methods=["POST"])
def api_gpu_url():
    """Met à jour l'URL ngrok sans redémarrer Render"""
    data    = request.get_json(force=True)
    new_url = data.get("url", "").strip()
    if not new_url.startswith("http"):
        return jsonify({"error": "URL invalide"}), 400
    os.environ["COLAB_URL_PRIMARY"] = new_url
    log.info(f"[CONFIG] URL GPU mise à jour: {new_url}")
    return jsonify({"ok": True, "url": new_url})


@app.route("/api/health", methods=["GET"])
def api_health():
    platforms = get_platforms()
    gpu_status = {}
    for name, url in platforms.items():
        if not url.startswith("http"):
            gpu_status[name] = "non_configuré"
            continue
        try:
            r = requests.get(url.rstrip("/") + "/health", timeout=8, headers=NGROK_HEADERS)
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
        "version":    "5.0.0",
        "queue_size": job_queue.qsize(),
        "jobs_total": len(job_status),
        "jobs_done":  sum(1 for j in job_status.values() if j["status"] == "done"),
        "gpu":        gpu_status,
    })


# ── Front-end static ─────────────────────────────────────────────────────────
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
    log.info("BENY-JOE CINÉ IA PRO v5.0 — http://localhost:5000")
    log.info(f"GPU PRIMARY: {get_platforms()['primary'][:70]}")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


@app.route("/api/pending_job", methods=["GET"])
def api_pending_job():
    for job_id, status in job_status.items():
        if status.get("status") == "queued":
            return jsonify(status)
    return jsonify({})
