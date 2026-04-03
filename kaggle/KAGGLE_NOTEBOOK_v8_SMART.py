# ╔══════════════════════════════════════════════════════════════════════╗
# ║   BENY-JOE CINÉ IA PRO v8.0 — NOTEBOOK INTELLIGENT                 ║
# ║   Fondé par KHEDIM BENYAKHLEF dit BENY-JOE                          ║
# ║                                                                      ║
# ║   STRATÉGIE v8 — ZÉRO GASPILLAGE GPU :                              ║
# ║   ✅ CPU actif en permanence (génération lente mais RÉELLE)          ║
# ║   ✅ GPU chargé UNIQUEMENT quand un job arrive                       ║
# ║   ✅ GPU libéré (unload) après chaque job → VRAM récupérée           ║
# ║   ✅ Auto-extinction après 20 min d'inactivité                       ║
# ║   ✅ CPU mode = vidéos courtes réelles (4 frames, 256x256)           ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════
# CELLULE 1 — Installation
# ══════════════════════════════════════════════════════════════════════
import subprocess

def pip(pkg):
    r = subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True, text=True)
    print(f"  {pkg[:50]:<50} {'OK' if r.returncode == 0 else 'WARN'}")

print("=== Installation BENY-JOE STUDIO v8.0 ===")
pip("huggingface_hub==0.21.4")
pip("accelerate==0.27.2")
pip("diffusers==0.25.0")
pip("transformers==4.38.2")
pip("peft==0.9.0")
pip("omegaconf einops safetensors")
pip("imageio==2.33.1 imageio-ffmpeg")
pip("Pillow>=10.0.0 opencv-python-headless")
pip("coqui-tts")
pip("moviepy==1.0.3 ffmpeg-python pydub scipy")
pip("flask==3.0.3 flask-cors==4.0.1 requests")
print("\n✅ Installation terminée!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 2 — Détection hardware
# ══════════════════════════════════════════════════════════════════════
import torch, os, uuid as _uuid, random, threading, time, logging, gc
import base64, io, numpy as np, imageio
from PIL import Image

device  = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = device == "cuda"

if use_gpu:
    vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} | {vram} GB VRAM disponible")
else:
    vram = 0
    print("⚠️ CPU détecté — génération RÉELLE mais plus lente (3-5 min par vidéo)")

BASE_WORK  = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
OUTPUT_DIR = os.path.join(BASE_WORK, "outputs")
FRAMES_DIR = os.path.join(BASE_WORK, "frames")
AUDIO_DIR  = os.path.join(BASE_WORK, "audio")
TEMP_DIR   = os.path.join(BASE_WORK, "temp")
for d in [OUTPUT_DIR, FRAMES_DIR, AUDIO_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("BenyJoe")

# ── Paramètres selon le mode ──────────────────────────────────────────
if use_gpu:
    dtype = torch.float16
    DEFAULT_FRAMES, DEFAULT_STEPS, DEFAULT_W, DEFAULT_H, DEFAULT_FPS, CHUNK_SIZE = 24, 25, 512, 512, 8, 8
else:
    dtype = torch.float32
    # CPU : frames réduites pour que ça se TERMINE en temps raisonnable
    DEFAULT_FRAMES, DEFAULT_STEPS, DEFAULT_W, DEFAULT_H, DEFAULT_FPS, CHUNK_SIZE = 4, 6, 256, 256, 4, 2

print(f"Mode: {'GPU 🚀' if use_gpu else 'CPU ⚙️ (réel)'} | {DEFAULT_W}x{DEFAULT_H} | {DEFAULT_FRAMES}f | {DEFAULT_STEPS} steps")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 3 — État global + gestion VRAM intelligente
# ══════════════════════════════════════════════════════════════════════
_state = {
    "running": False, "progress": 0, "step": "En attente...",
    "current_frame": 0, "total_frames": 0,
    "video_path": None, "image_path": None,
    "audio_path": None, "final_path": None,
    "error": None, "job_id": None,
    "last_job_time": time.time(),  # ← pour auto-extinction
}
_lock = threading.Lock()

# Modèles — None = pas chargé (chargé à la demande)
pipe_video   = None
pipe_image   = None
pipe_img2vid = None
tts_model    = None
tts_available = False
music_available = False

IDLE_TIMEOUT = 20 * 60  # 20 minutes sans job → extinction

def reset_state(total_frames=0, job_id=None):
    with _lock:
        _state.update({
            "running": True, "progress": 0, "step": "Initialisation...",
            "current_frame": 0, "total_frames": total_frames,
            "video_path": None, "image_path": None,
            "audio_path": None, "final_path": None,
            "error": None, "job_id": job_id or _uuid.uuid4().hex[:8],
            "last_job_time": time.time(),
        })

def update_state(**kw):
    with _lock:
        _state.update(kw)

def touch_activity():
    """Réinitialise le timer d'inactivité."""
    with _lock:
        _state["last_job_time"] = time.time()

def free_vram():
    """Libère la VRAM après chaque job."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def unload_models():
    """Décharge tous les modèles de la VRAM — économise la VRAM entre les jobs."""
    global pipe_video, pipe_image, pipe_img2vid
    if pipe_video   is not None: del pipe_video;   pipe_video   = None
    if pipe_image   is not None: del pipe_image;   pipe_image   = None
    if pipe_img2vid is not None: del pipe_img2vid; pipe_img2vid = None
    free_vram()
    log.info("[VRAM] Modèles déchargés — VRAM libérée")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 4 — Chargement modèles À LA DEMANDE (pas au démarrage)
# ══════════════════════════════════════════════════════════════════════
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_video

def load_video_model():
    """Charge AnimateDiff uniquement quand nécessaire."""
    global pipe_video
    if pipe_video is not None:
        return pipe_video  # déjà chargé

    log.info(f"[LOAD] Chargement AnimateDiff en mode {device}...")
    update_state(step="⏳ Chargement modèle vidéo...")

    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
    pipe_video = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter, torch_dtype=dtype, low_cpu_mem_usage=True)
    pipe_video.scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler",
        clip_sample=False, timestep_spacing="linspace",
        beta_schedule="linear", steps_offset=1)
    pipe_video = pipe_video.to(device)
    pipe_video.vae.enable_slicing()
    pipe_video.enable_attention_slicing()
    try: pipe_video.enable_vae_tiling()
    except: pass

    log.info(f"[LOAD] AnimateDiff prêt | {device}")
    return pipe_video

# TTS — chargé une fois, léger, on le garde en mémoire
def init_tts():
    global tts_model, tts_available
    if tts_available:
        return
    try:
        from TTS.api import TTS as CoquiTTS
        os.environ["COQUI_TOS_AGREED"] = "1"
        tts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        tts_available = True
        log.info("✅ TTS prêt")
    except Exception as e:
        log.warning(f"TTS non dispo: {str(e)[:60]}")

# MusicGen — optionnel, skip sur CPU (trop lent)
def init_music():
    global music_available
    if not use_gpu:
        log.info("MusicGen skippé en mode CPU (trop lent)")
        return
    try:
        from audiocraft.models import MusicGen
        globals()["music_model"] = MusicGen.get_pretrained("facebook/musicgen-small")
        globals()["music_model"].set_generation_params(duration=10)
        music_available = True
        log.info("✅ MusicGen prêt")
    except Exception as e:
        log.warning(f"MusicGen non dispo: {str(e)[:60]}")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 5 — Génération vidéo (CPU: réelle mais courte)
# ══════════════════════════════════════════════════════════════════════
STYLE_PROMPTS = {
    "cinematic":  "cinematic film, professional lighting, high quality, 4K",
    "cyberpunk":  "cyberpunk neon lights, futuristic city, blade runner style",
    "epic":       "epic fantasy, dramatic lighting, cinematic scope",
    "nature":     "nature documentary, realistic, wildlife photography",
    "scifi":      "science fiction, space opera, futuristic technology",
    "filmnoir":   "film noir, black and white, 1940s detective, dramatic shadows",
    "fantasy":    "fantasy magical world, ethereal lighting, enchanted",
    "horror":     "horror atmosphere, dark shadows, suspense, eerie lighting",
    "romantique": "romantic soft lighting, golden hour, emotional, beautiful",
}
NEG_PROMPT = "blurry, low quality, bad anatomy, deformed, ugly, duplicate, error, jpeg artifacts, watermark, text, cartoon, anime"

def generate_video_from_text(prompt, style="cinematic", num_frames=None,
        width=None, height=None, steps=None, guidance_scale=7.5,
        fps=None, seed=-1):
    pipe = load_video_model()

    nf = num_frames or DEFAULT_FRAMES
    w  = width      or DEFAULT_W
    h  = height     or DEFAULT_H
    st = steps      or DEFAULT_STEPS
    fp = fps        or DEFAULT_FPS

    # Sur CPU on force des valeurs raisonnables
    if not use_gpu:
        nf = min(nf, 6)
        w  = min(w,  256)
        h  = min(h,  256)
        st = min(st, 8)

    full_prompt = f"{prompt}, {STYLE_PROMPTS.get(style, '')}"
    generator   = torch.Generator(device=device).manual_seed(seed) if seed >= 0 else None

    update_state(step=f"🎨 Génération vidéo {'GPU 🚀' if use_gpu else 'CPU ⚙️'}... ({nf} frames, {w}x{h}, {st} steps)")
    log.info(f"Vidéo: {nf}f {w}x{h} {st}steps | {'GPU' if use_gpu else 'CPU'}")

    total_steps = [0]
    def cb(pipe, step, timestep, kwargs):
        pct = 10 + int((step / st) * 75)
        update_state(progress=pct, step=f"🎨 Step {step}/{st}...", current_frame=step, total_frames=st)
        return kwargs

    out = pipe(prompt=full_prompt, negative_prompt=NEG_PROMPT,
               num_frames=nf, width=w, height=h,
               num_inference_steps=st, guidance_scale=guidance_scale,
               generator=generator, callback_on_step_end=cb)

    vid_path = os.path.join(OUTPUT_DIR, f"video_{_uuid.uuid4().hex[:8]}.mp4")
    export_to_video(out.frames[0], vid_path, fps=fp)
    update_state(video_path=vid_path, progress=85, step="✅ Vidéo générée")
    free_vram()
    return vid_path, os.path.basename(vid_path), out.frames[0]

# ══════════════════════════════════════════════════════════════════════
# CELLULE 6 — Voix off + assemblage final
# ══════════════════════════════════════════════════════════════════════
VOICE_PROFILES = {
    "masculin":   {"speaker": "Claribel Dervla",  "language": "fr", "speed": 0.9},
    "feminin":    {"speaker": "Daisy Studious",    "language": "fr", "speed": 1.0},
    "narrateur":  {"speaker": "Gracie Wise",       "language": "fr", "speed": 0.85},
    "dynamique":  {"speaker": "Tammie Ema",        "language": "fr", "speed": 1.1},
}

def generate_voix_off(texte, style_voix="masculin", output_path=None):
    if not tts_available or not texte:
        return None
    try:
        profile = VOICE_PROFILES.get(style_voix, VOICE_PROFILES["masculin"])
        out_path = output_path or os.path.join(AUDIO_DIR, f"voix_{_uuid.uuid4().hex[:8]}.wav")
        tts_model.tts_to_file(text=texte, speaker=profile["speaker"],
                              language=profile["language"],
                              speed=profile.get("speed", 1.0), file_path=out_path)
        return out_path if os.path.exists(out_path) else None
    except Exception as e:
        log.warning(f"TTS: {e}")
        return None

def generate_musique(style="cinematique", duree=10, output_path=None):
    if not music_available:
        return None
    try:
        music_prompts = {
            "cinematique": "epic cinematic orchestral music, dramatic",
            "action":      "fast action movie soundtrack, intense drums",
            "romantique":  "romantic soft piano music, emotional",
            "horreur":     "horror suspense music, dark ambient",
            "cyberpunk":   "cyberpunk electronic music, synthwave",
        }
        m = globals().get("music_model")
        if not m: return None
        m.set_generation_params(duration=duree)
        wav = m.generate([music_prompts.get(style, "background music")])
        out = output_path or os.path.join(AUDIO_DIR, f"music_{_uuid.uuid4().hex[:8]}.wav")
        import torchaudio
        torchaudio.save(out, wav[0].cpu(), 32000)
        return out if os.path.exists(out) else None
    except Exception as e:
        log.warning(f"Music: {e}")
        return None

def assemble_final_video(video_path, job_id, prompt="",
        style_video="cinematic", voix_active=True, style_voix="masculin",
        texte_voix=None, musique_active=True, style_musique="cinematique",
        volume_voix=0.9, volume_musique=0.28):
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

    update_state(progress=87, step="🎵 Audio...")

    if not os.path.exists(video_path):
        update_state(running=False, error="Fichier vidéo introuvable", step="❌ Erreur")
        return

    texte = texte_voix or prompt[:200]
    voix_path  = generate_voix_off(texte, style_voix) if voix_active else None
    music_path = generate_musique(style_musique) if musique_active else None

    update_state(progress=92, step="🎞️ Assemblage final...")

    try:
        clip     = VideoFileClip(video_path)
        duration = clip.duration
        audio_clips = []

        if music_path and os.path.exists(music_path):
            m = AudioFileClip(music_path).subclip(0, min(duration, AudioFileClip(music_path).duration))
            audio_clips.append(m.volumex(volume_musique))

        if voix_path and os.path.exists(voix_path):
            v = AudioFileClip(voix_path).subclip(0, min(duration, AudioFileClip(voix_path).duration))
            audio_clips.append(v.volumex(volume_voix))

        final_path = os.path.join(OUTPUT_DIR, f"BENYJOE_FINAL_{job_id}.mp4")

        if audio_clips:
            mixed = CompositeAudioClip(audio_clips)
            final_clip = clip.set_audio(mixed)
        else:
            final_clip = clip

        final_clip.write_videofile(final_path, codec="libx264",
                                   audio_codec="aac", verbose=False, logger=None)
        clip.close()

        update_state(running=False, progress=100, step="✅ Terminé!",
                     final_path=final_path, video_path=final_path)
        log.info(f"✅ Final: {final_path}")

    except Exception as e:
        log.exception("Assemblage")
        update_state(running=False, error=str(e), step="❌ Erreur assemblage")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 7 — Flask GPU server
# ══════════════════════════════════════════════════════════════════════
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

gpu_app = Flask(__name__)
CORS(gpu_app)

def run_async(fn):
    def _run():
        try:
            fn()
        except Exception as e:
            update_state(running=False, error=str(e), step="❌ Erreur")
            log.exception("Erreur génération")
    threading.Thread(target=_run, daemon=True).start()

@gpu_app.route("/health")
def health():
    touch_activity()
    idle_since = int(time.time() - _state["last_job_time"])
    return jsonify({
        "status": "ok", "device": device, "vram_gb": vram,
        "running": _state["running"], "progress": _state["progress"],
        "step": _state["step"], "version": "8.0.0",
        "current_frame": _state["current_frame"],
        "total_frames": _state["total_frames"],
        "mode": "GPU" if use_gpu else "CPU",
        "idle_seconds": idle_since,
        "models_loaded": pipe_video is not None,
    })

@gpu_app.route("/progress")
def prog():
    return jsonify({
        "progress": _state["progress"], "running": _state["running"],
        "step": _state["step"], "current_frame": _state["current_frame"],
        "total_frames": _state["total_frames"], "video_path": _state["video_path"],
        "image_path": _state["image_path"], "final_path": _state["final_path"],
        "error": _state["error"],
    })

@gpu_app.route("/generate", methods=["POST"])
def api_generate():
    if _state["running"]:
        return jsonify({"error": "Génération en cours — patientez"}), 429
    d      = request.get_json(force=True)
    prompt = d.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt vide"}), 400

    touch_activity()
    job_id = _uuid.uuid4().hex[:8]
    nf     = min(int(d.get("num_frames", DEFAULT_FRAMES)), 40 if use_gpu else 6)
    reset_state(total_frames=nf, job_id=job_id)

    def _gen():
        vp, vn, _ = generate_video_from_text(
            prompt=prompt, style=d.get("style", "cinematic"),
            num_frames=nf,
            width=int(d.get("width", DEFAULT_W)),
            height=int(d.get("height", DEFAULT_H)),
            steps=min(int(d.get("steps", DEFAULT_STEPS)), 35 if use_gpu else 8),
            guidance_scale=float(d.get("guidance_scale", 7.5)),
            fps=int(d.get("fps", DEFAULT_FPS)),
            seed=int(d.get("seed", -1)))
        assemble_final_video(vp, job_id, prompt=prompt,
            style_video=d.get("style", "cinematic"),
            voix_active=bool(d.get("voix_active", True)) and tts_available,
            style_voix=d.get("style_voix", "masculin"),
            texte_voix=d.get("texte_voix", None),
            musique_active=bool(d.get("musique_active", True)) and music_available,
            style_musique=d.get("style_musique", "cinematique"),
            volume_voix=float(d.get("volume_voix", 0.9)),
            volume_musique=float(d.get("volume_musique", 0.28)))
        # ← Décharger les modèles après le job pour libérer la VRAM
        if use_gpu:
            unload_models()

    run_async(_gen)
    return jsonify({"status": "started", "job_id": job_id,
                    "mode": "GPU 🚀" if use_gpu else "CPU ⚙️ (réel)"})

@gpu_app.route("/generate_image", methods=["POST"])
def api_generate_image():
    if _state["running"]:
        return jsonify({"error": "Occupé"}), 429
    d      = request.get_json(force=True)
    prompt = d.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt vide"}), 400

    touch_activity()
    job_id = _uuid.uuid4().hex[:8]
    reset_state(job_id=job_id)

    def _gen():
        from diffusers import StableDiffusionPipeline
        update_state(step="⏳ Chargement modèle image...")
        p = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=dtype).to(device)
        update_state(step="🎨 Génération image...")
        img = p(prompt, num_inference_steps=int(d.get("steps", 25)),
                guidance_scale=float(d.get("guidance_scale", 7.5))).images[0]
        out = os.path.join(OUTPUT_DIR, f"img_{job_id}.png")
        img.save(out)
        del p; free_vram()
        update_state(running=False, progress=100, step="✅ Image prête!", image_path=out)
        if use_gpu: unload_models()

    run_async(_gen)
    return jsonify({"status": "started", "job_id": job_id})

@gpu_app.route("/video/<f>")
def srv_v(f):
    p = os.path.join(OUTPUT_DIR, f)
    return send_file(p, mimetype="video/mp4") if os.path.exists(p) else ("Not found", 404)

@gpu_app.route("/final/<f>")
def srv_f(f):
    p = os.path.join(OUTPUT_DIR, f)
    return send_file(p, mimetype="video/mp4") if os.path.exists(p) else ("Not found", 404)

@gpu_app.route("/image/<f>")
def srv_i(f):
    p = os.path.join(OUTPUT_DIR, f)
    return send_file(p, mimetype="image/png") if os.path.exists(p) else ("Not found", 404)

@gpu_app.route("/audio/<f>")
def srv_a(f):
    p = os.path.join(AUDIO_DIR, f)
    return send_file(p, mimetype="audio/wav") if os.path.exists(p) else ("Not found", 404)

@gpu_app.route("/list_outputs")
def list_out():
    files = [{"name": f, "size_mb": round(os.path.getsize(os.path.join(OUTPUT_DIR, f))/1e6, 2)}
             for f in os.listdir(OUTPUT_DIR) if not f.startswith(".")]
    return jsonify({"files": sorted(files, key=lambda x: x["name"], reverse=True)})

print("✅ Flask GPU — routes définies!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 8 — TTS init (léger, fait au démarrage)
# ══════════════════════════════════════════════════════════════════════
threading.Thread(target=init_tts, daemon=True).start()
# MusicGen ne se charge QUE si GPU
if use_gpu:
    threading.Thread(target=init_music, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════
# CELLULE 9 — Auto-extinction après inactivité
# ══════════════════════════════════════════════════════════════════════
def auto_shutdown_watcher():
    """Éteint le notebook après IDLE_TIMEOUT secondes sans activité."""
    while True:
        time.sleep(60)
        idle = time.time() - _state["last_job_time"]
        remaining = int((IDLE_TIMEOUT - idle) / 60)
        if idle > IDLE_TIMEOUT:
            log.warning(f"[AUTO-STOP] {IDLE_TIMEOUT//60} min d'inactivité → extinction pour économiser GPU")
            # Libère la VRAM
            unload_models()
            # Signal propre au notebook Kaggle
            try:
                import IPython
                IPython.get_ipython().kernel.do_shutdown(False)
            except:
                os._exit(0)
        else:
            log.info(f"[IDLE] En attente depuis {int(idle//60)}min — extinction dans {remaining}min")

threading.Thread(target=auto_shutdown_watcher, daemon=True).start()
print(f"✅ Auto-extinction dans {IDLE_TIMEOUT//60} min d'inactivité")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 10 — Lancement Flask + ngrok
# ══════════════════════════════════════════════════════════════════════
import subprocess, requests

# ⚠️ MODIFIEZ CES 2 VALEURS !
NGROK_TOKEN = "VOTRE_TOKEN_NGROK_ICI"
RENDER_URL  = "https://benyjoe-studio.onrender.com"

subprocess.run(["pkill", "-f", "ngrok"],  capture_output=True)
subprocess.run(["pkill", "-f", "flask"],  capture_output=True)
subprocess.run(["fuser", "-k", "5001/tcp"], capture_output=True)
time.sleep(3)

def run_flask():
    gpu_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)

threading.Thread(target=run_flask, daemon=True).start()
time.sleep(4)

# Vérifier Flask
flask_ok = False
for i in range(10):
    try:
        r = requests.get("http://127.0.0.1:5001/health", timeout=3)
        if r.status_code == 200:
            d = r.json()
            print(f"✅ Flask OK | {d['device']} | VRAM:{d['vram_gb']}GB | mode:{d.get('mode','?')}")
            flask_ok = True
            break
    except:
        pass
    time.sleep(2)

if not flask_ok:
    print("❌ Flask ne répond pas — relancez cette cellule")
else:
    # Install + lancer ngrok
    try:
        subprocess.run(["ngrok", "version"], capture_output=True, check=True)
    except:
        print("Installation ngrok...")
        subprocess.run(
            "curl -sL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | "
            "tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && "
            "echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | "
            "tee /etc/apt/sources.list.d/ngrok.list >/dev/null && "
            "apt-get update -qq && apt-get install -qq ngrok", shell=True)

    subprocess.run(["ngrok", "config", "add-authtoken", NGROK_TOKEN], capture_output=True)
    subprocess.Popen(["ngrok", "http", "5001"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)

    NGROK_URL = ""
    for i in range(12):
        try:
            r = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
            for t in r.json().get("tunnels", []):
                if t.get("proto") == "https":
                    NGROK_URL = t["public_url"]
                    break
            if NGROK_URL:
                break
        except:
            pass
        time.sleep(2)
        print(f"  Attente ngrok {i+1}/12...")

    if NGROK_URL:
        print(f"\n{'='*60}")
        print(f"  BENY-JOE CINÉ IA PRO v8.0 — PRÊT!")
        print(f"  Mode    : {'GPU 🚀 (déchargé après chaque job)' if use_gpu else 'CPU ⚙️ (réel, 3-5 min/vidéo)'}")
        print(f"  VRAM    : {vram} GB | Auto-extinction: {IDLE_TIMEOUT//60} min")
        print(f"  URL     : {NGROK_URL}")
        print(f"{'='*60}")

        # Auto-update Render
        try:
            r = requests.post(f"{RENDER_URL}/api/gpu_url",
                json={"url": NGROK_URL}, timeout=15)
            if r.json().get("ok"):
                print("  ✅ Render mis à jour automatiquement!")
            else:
                print(f"  → Copiez dans Render: {NGROK_URL}")
        except:
            print(f"  → Copiez dans Render → COLAB_URL_PRIMARY: {NGROK_URL}")

        # Keep-alive intelligent — ping moins souvent
        def keep_alive():
            while True:
                try:
                    r = requests.get(NGROK_URL + "/health", timeout=10,
                        headers={"ngrok-skip-browser-warning": "true"})
                    d = r.json()
                    idle_min = int(d.get("idle_seconds", 0) // 60)
                    status = f"{d['step']} ({d['progress']}%)" if d.get("running") \
                             else f"Prêt | {d['mode']} | idle:{idle_min}min"
                    print(f"  ♥ {status}")
                except Exception as e:
                    print(f"  Ping err: {e}")
                time.sleep(120)

        threading.Thread(target=keep_alive, daemon=True).start()
        print(f"  Keep-alive actif — Studio opérationnel! 🎬")
    else:
        print("❌ URL ngrok non trouvée — vérifiez votre token NGROK")
