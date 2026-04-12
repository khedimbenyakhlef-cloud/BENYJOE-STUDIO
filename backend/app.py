"""
BENY-JOE CINÉ IA PRO — Backend Flask v11.0
Mode PUSH : Colab poll /api/pending_job et push /api/save_result
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, logging, threading, hashlib
from datetime import datetime

app  = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FRONT_DIR = os.path.join(BASE_DIR, "..", "frontend")
LOG_DIR   = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "server.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("BenyJoeStudio")

_RAW_PIN        = os.getenv("ACCESS_PIN", "2022002")
ACCESS_PIN_HASH = hashlib.sha256(_RAW_PIN.encode()).hexdigest()

def check_pin(p):
    return hashlib.sha256(str(p).encode()).hexdigest() == ACCESS_PIN_HASH

_gpu_url = {"url": os.getenv("COLAB_URL_PRIMARY", "NOT_CONFIGURED")}
def get_gpu_url(): return _gpu_url["url"]
def set_gpu_url(u):
    _gpu_url["url"] = u
    os.environ["COLAB_URL_PRIMARY"] = u

NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json",
}

job_status   = {}
pending_jobs = {}
_lock        = threading.Lock()

@app.route("/api/auth", methods=["POST"])
def api_auth():
    d = request.get_json(force=True)
    if check_pin(str(d.get("pin",""))):
        return jsonify({"ok": True, "version": "11.0.0"})
    return jsonify({"ok": False, "error": "PIN incorrect"}), 401

@app.route("/api/generate", methods=["POST"])
def api_generate():
    d      = request.get_json(force=True)
    prompt = d.get("prompt","").strip()
    if not prompt:
        return jsonify({"error": "Prompt vide"}), 400
    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id":         job_id,
        "type":           "video",
        "prompt":         prompt,
        "style":          d.get("style","cinematic"),
        "num_frames":     min(int(d.get("frames", 80)), 120),
        "width":          int(d.get("width", 512)),
        "height":         int(d.get("height", 512)),
        "steps":          min(int(d.get("steps", 30)), 50),
        "guidance_scale": float(d.get("guidance", 9.0)),
        "fps":            int(d.get("fps", 16)),
        "seed":           int(d.get("seed", -1)),
        "voix_active":    bool(d.get("voix_active", True)),
        "style_voix":     d.get("style_voix", "masculin"),
        "texte_voix":     d.get("texte_voix", None),
        "musique_active": bool(d.get("musique_active", True)),
        "style_musique":  d.get("style_musique", "cinematique"),
        "volume_voix":    float(d.get("volume_voix", 0.9)),
        "volume_musique": float(d.get("volume_musique", 0.28)),
        "created_at":     datetime.now().isoformat(),
    }
    with _lock:
        job_status[job_id] = {
            "type":       "video",
            "status":     "queued",
            "result":     None,
            "error":      None,
            "progress":   0,
            "step":       "En attente Colab...",
            "created_at": job["created_at"],
            "prompt":     prompt,
            "style":      job["style"],
        }
        pending_jobs[job_id] = job
    log.info(f"[QUEUE] {job_id} | {prompt[:50]}")
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route("/api/generate_image", methods=["POST"])
def api_generate_image():
    d      = request.get_json(force=True)
    prompt = d.get("prompt","").strip()
    if not prompt:
        return jsonify({"error": "Prompt vide"}), 400
    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id":     job_id,
        "type":       "image",
        "prompt":     prompt,
        "style":      d.get("style","cinematic"),
        "resolution": d.get("resolution","1024x1024"),
        "steps":      min(int(d.get("steps",30)),50),
        "guidance_scale": float(d.get("guidance",7.5)),
        "seed":       int(d.get("seed",-1)),
        "created_at": datetime.now().isoformat(),
    }
    with _lock:
        job_status[job_id] = {
            "type":"image","status":"queued","result":None,"error":None,
            "progress":0,"step":"En attente Colab...","created_at":job["created_at"],
            "prompt":prompt,"style":job["style"],
        }
        pending_jobs[job_id] = job
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route("/api/pending_job", methods=["GET"])
def api_pending_job():
    with _lock:
        for job_id, job in list(pending_jobs.items()):
            job_status[job_id].update({
                "status":   "processing",
                "step":     "Envoyé à Colab...",
                "progress": 5,
            })
            del pending_jobs[job_id]
            log.info(f"[DISPATCH] {job_id} → Colab")
            return jsonify(job)
    return jsonify({})

@app.route("/api/save_result", methods=["POST"])
def api_save_result():
    d      = request.get_json(force=True)
    job_id = d.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id manquant"}), 400
    with _lock:
        if job_id not in job_status:
            job_status[job_id] = {
                "type":       "video",
                "created_at": datetime.now().isoformat(),
                "prompt":     d.get("prompt","Généré depuis Colab"),
                "style":      "cinematic",
            }
        job_status[job_id].update({
            "status":   d.get("status","done"),
            "result":   d.get("result"),
            "progress": int(d.get("progress",100)),
            "step":     d.get("step","✅ Terminé!"),
            "error":    d.get("error"),
        })
    log.info(f"[RESULT] {job_id} → {str(d.get('result',''))[:60]}")
    return jsonify({"ok": True, "job_id": job_id})

@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    job = job_status.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable"}), 404
    return jsonify(job)

@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id):
    if job_id in job_status:
        job_status[job_id].update({"status":"cancelled","error":"Annulé"})
        pending_jobs.pop(job_id, None)
        return jsonify({"ok": True})
    return jsonify({"error": "Job introuvable"}), 404

@app.route("/api/clear_queue", methods=["POST"])
def api_clear_queue():
    with _lock:
        pending_jobs.clear()
        for j in job_status.values():
            if j["status"] == "queued":
                j.update({"status":"cancelled","error":"File vidée"})
    return jsonify({"ok": True})

@app.route("/api/history", methods=["GET"])
def api_history():
    jobs = sorted(job_status.values(),
                  key=lambda x: x.get("created_at",""), reverse=True)
    return jsonify({"jobs": jobs[:50]})

@app.route("/api/gpu_url", methods=["POST"])
def api_gpu_url():
    d = request.get_json(force=True)
    u = d.get("url","").strip()
    if not u.startswith("http"):
        return jsonify({"error": "URL invalide"}), 400
    set_gpu_url(u)
    return jsonify({"ok": True, "url": u})

@app.route("/api/health", methods=["GET"])
def api_health():
    import requests as req
    gpu_url = get_gpu_url()
    gpu_st  = "non_configuré"
    if gpu_url.startswith("http"):
        try:
            r  = req.get(gpu_url.rstrip("/")+"/health", timeout=8, headers=NGROK_HEADERS)
            ct = r.headers.get("Content-Type","")
            if "html" not in ct.lower():
                d      = r.json()
                gpu_st = {
                    "status":  "online" if r.status_code==200 else "error",
                    "device":  d.get("device","?"),
                    "vram_gb": d.get("vram_gb",0),
                    "running": d.get("running",False),
                    "step":    d.get("step",""),
                    "progress":d.get("progress",0),
                }
            else:
                gpu_st = {"status":"html_error"}
        except Exception as e:
            gpu_st = {"status":"offline","error":str(e)[:80]}

    return jsonify({
        "status":     "ok",
        "version":    "11.0.0",
        "pending":    len(pending_jobs),
        "jobs_total": len(job_status),
        "jobs_done":  sum(1 for j in job_status.values() if j["status"]=="done"),
        "gpu":        {"primary": gpu_st},
        "gpu_url":    gpu_url,
    })

@app.route("/")
def index():
    return send_from_directory(FRONT_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONT_DIR, path)

if __name__ == "__main__":
    log.info("BENY-JOE CINÉ IA PRO v11.0")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
