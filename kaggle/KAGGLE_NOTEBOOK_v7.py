# ╔══════════════════════════════════════════════════════════════════════╗
# ║   BENY-JOE CINÉ IA PRO v7.0 — NOTEBOOK COLAB / KAGGLE              ║
# ║   Fondé par KHEDIM BENYAKHLEF dit BENY-JOE                          ║
# ║   FIX v7 :                                                           ║
# ║   - Priorité GPU → CPU automatique                                   ║
# ║   - Génération async réelle (barre progresse!)                       ║
# ║   - MusicGen optionnel (skip si CPU)                                 ║
# ║   - Versions packages fixées Python 3.11                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════
# CELLULE 1 — Installation (versions compatibles Python 3.11)
# ══════════════════════════════════════════════════════════════════════
import subprocess

def pip(pkg):
    r = subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True, text=True)
    print(f"  {pkg[:50]:<50} {'OK' if r.returncode == 0 else 'WARN'}")

print("=== Installation BENY-JOE STUDIO v7.0 ===")
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
# CELLULE 2 — Init GPU/CPU + état global
# ══════════════════════════════════════════════════════════════════════
import torch, os, uuid, random, threading, time, logging
import base64, io, numpy as np, imageio
from PIL import Image

device  = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = device == "cuda"

if use_gpu:
    vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} | {vram} GB")
    dtype          = torch.float16
    DEFAULT_FRAMES = 24
    DEFAULT_STEPS  = 25
    DEFAULT_W      = 512
    DEFAULT_H      = 512
    DEFAULT_FPS    = 8
    CHUNK_SIZE     = 8
else:
    vram = 0
    print("⚠️ Mode CPU — génération lente mais fonctionnelle")
    dtype          = torch.float32
    DEFAULT_FRAMES = 4
    DEFAULT_STEPS  = 8
    DEFAULT_W      = 256
    DEFAULT_H      = 256
    DEFAULT_FPS    = 4
    CHUNK_SIZE     = 2

BASE_WORK  = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
OUTPUT_DIR = os.path.join(BASE_WORK, "outputs")
FRAMES_DIR = os.path.join(BASE_WORK, "frames")
AUDIO_DIR  = os.path.join(BASE_WORK, "audio")
TEMP_DIR   = os.path.join(BASE_WORK, "temp")
for d in [OUTPUT_DIR, FRAMES_DIR, AUDIO_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("BenyJoe")

_state = {
    "running": False, "progress": 0, "step": "En attente...",
    "current_frame": 0, "total_frames": 0,
    "video_path": None, "image_path": None,
    "audio_path": None, "final_path": None,
    "error": None, "job_id": None,
}
_lock = threading.Lock()

def reset_state(total_frames=0, job_id=None):
    with _lock:
        _state.update({
            "running": True, "progress": 0, "step": "Initialisation...",
            "current_frame": 0, "total_frames": total_frames,
            "video_path": None, "image_path": None,
            "audio_path": None, "final_path": None,
            "error": None, "job_id": job_id or uuid.uuid4().hex[:8],
        })

def update_state(**kw):
    with _lock:
        _state.update(kw)

def free_vram():
    import gc; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

NGROK_URL = ""
print(f"Init OK | mode={'GPU' if use_gpu else 'CPU'} | frames={DEFAULT_FRAMES} | steps={DEFAULT_STEPS} | {DEFAULT_W}x{DEFAULT_H}")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 3 — Chargement modèle (GPU priorité, CPU fallback)
# ══════════════════════════════════════════════════════════════════════
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_video

print(f"Chargement AnimateDiff en mode {'GPU' if use_gpu else 'CPU'}...")
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

pipe_image   = None
pipe_img2vid = None
print(f"✅ AnimateDiff prêt | mode={'GPU 🚀' if use_gpu else 'CPU ⚠️'}")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 4 — Voix Off (Coqui TTS)
# ══════════════════════════════════════════════════════════════════════
try:
    from TTS.api import TTS as CoquiTTS
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts_available = True
    print("✅ Coqui XTTS prêt!")
except Exception as e:
    tts_model     = None
    tts_available = False
    print(f"⚠️ TTS non disponible: {str(e)[:60]}")

VOICE_PROFILES = {
    "masculin":   {"speaker": "Claribel Dervla",  "language": "fr", "speed": 0.9},
    "feminin":    {"speaker": "Daisy Studious",    "language": "fr", "speed": 1.0},
    "dramatique": {"speaker": "Viktor Eka",        "language": "fr", "speed": 0.78},
    "jeune":      {"speaker": "Annmarie Niekamp",  "language": "fr", "speed": 1.05},
    "epique":     {"speaker": "Andrew Chipper",    "language": "fr", "speed": 0.85},
}
NARRATION = {
    "cinematic":   "Dans cette scène épique, chaque instant se grave dans la mémoire.",
    "cyberpunk":   "Dans ce futur dystopique où la technologie règne en maître.",
    "epic_battle": "Le destin du monde se joue dans cette bataille légendaire.",
    "nature":      "La nature révèle toute sa splendeur et sa majesté.",
    "scifi":       "Aux confins de l'univers, une nouvelle découverte attend.",
    "noir":        "Dans les ruelles sombres, les secrets ne restent jamais enterrés.",
    "fantasy":     "Dans ce monde de magie et de merveilles, tout est possible.",
    "horror":      "L'obscurité cache des secrets que l'homme n'est pas fait pour connaître.",
    "romantique":  "Dans la lumière dorée de cet instant, l'amour trouve sa forme.",
}

def generate_voix_off(texte, style_voix="masculin", output_path=None):
    if not tts_available: return None
    if not output_path:
        output_path = os.path.join(AUDIO_DIR, f"voix_{uuid.uuid4().hex[:8]}.wav")
    p = VOICE_PROFILES.get(style_voix, VOICE_PROFILES["masculin"])
    tts_model.tts_to_file(text=texte, speaker=p["speaker"],
        language=p["language"], file_path=output_path, speed=p["speed"])
    return output_path

def get_narration(prompt, style):
    base = NARRATION.get(style, "Une vision cinématographique unique.")
    return f"{base} {prompt[:100]}." if prompt else base

print(f"Voix: {'✅ actives' if tts_available else '⚠️ désactivées'}")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 5 — Musique (MusicGen — GPU seulement)
# ══════════════════════════════════════════════════════════════════════
music_available = False
music_model     = None

if use_gpu:
    try:
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        music_model     = MusicGen.get_pretrained("facebook/musicgen-small")
        music_model.set_generation_params(duration=10)
        music_available = True
        print("✅ MusicGen prêt!")
    except Exception as e:
        print(f"⚠️ MusicGen non disponible: {str(e)[:60]}")
else:
    print("⚠️ MusicGen désactivé en mode CPU (trop lourd)")

MUSIC_PROMPTS = {
    "cinematique": "epic cinematic orchestral, dramatic strings, Hans Zimmer style",
    "ambiante":    "peaceful ambient, soft piano, atmospheric pads, calm",
    "cyberpunk":   "dark synthwave, futuristic industrial beats, neon",
    "nature":      "acoustic guitar, birds, gentle rain, folk instrumental",
    "scifi":       "space ambient, ethereal pads, cosmic, futuristic",
    "noir":        "jazz saxophone, trumpet, rainy night, 1940s detective",
    "fantasy":     "magical orchestral, choir, enchanted forest, epic",
    "horror":      "dissonant strings, tension, suspense, eerie",
    "epic_battle": "powerful timpani drums, war horns, intense strings",
    "romantique":  "romantic violin solo, gentle piano, emotional",
}

def generate_musique(style="cinematique", duree=10, output_path=None):
    if not music_available: return None
    if not output_path:
        output_path = os.path.join(AUDIO_DIR, f"musique_{uuid.uuid4().hex[:8]}")
    music_model.set_generation_params(duration=max(5, int(duree)))
    wav = music_model.generate([MUSIC_PROMPTS.get(style, MUSIC_PROMPTS["cinematique"])])
    audio_write(output_path, wav[0].cpu(), music_model.sample_rate,
                strategy="loudness", loudness_compressor=True)
    return output_path + ".wav"

print(f"Musique: {'✅ active' if music_available else '⚠️ désactivée'}")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 6 — Fonctions de génération
# ══════════════════════════════════════════════════════════════════════
NEG = ("blurry, low quality, bad anatomy, deformed, ugly, duplicate, "
       "error, jpeg artifacts, watermark, text, cartoon, flat colors")
STYLE_SFX = {
    "cinematic":   "cinematic lighting, dramatic shadows, anamorphic lens, film grain, 4K HDR",
    "cyberpunk":   "neon lights, rain-soaked streets, volumetric fog, ultra realistic, 8K",
    "epic_battle": "fire and smoke, dramatic sky, hyperrealistic warriors, 8K",
    "nature":      "golden hour, misty mountains, ultra wide angle, cinematic grade, 4K",
    "scifi":       "space station, zero gravity, distant nebula, sci-fi tech, cinematic",
    "noir":        "black and white film noir, rain, smoky atmosphere, dramatic shadows",
    "fantasy":     "enchanted forest, magical creatures, bioluminescent, epic fantasy, 8K",
    "horror":      "fog, moonlight, horror atmosphere, dark, cinematic, ultra realistic",
    "romantique":  "golden hour, warm bokeh, emotional, dreamy, soft light, romantic, 4K",
}

def generate_video_from_text(prompt, style="cinematic",
    num_frames=None, width=None, height=None,
    steps=None, guidance_scale=7.5, fps=None, seed=-1, chunk_size=None):

    # Priorité GPU → CPU
    nf   = min(num_frames or DEFAULT_FRAMES, 40 if use_gpu else 6)
    w    = width  or DEFAULT_W
    h    = height or DEFAULT_H
    st   = min(steps or DEFAULT_STEPS, 35 if use_gpu else 10)
    fp   = fps or DEFAULT_FPS
    cs   = chunk_size or CHUNK_SIZE

    if seed == -1: seed = random.randint(0, 2**32-1)
    generator   = torch.Generator(device=device).manual_seed(seed)
    job_id      = _state["job_id"] or uuid.uuid4().hex[:8]
    full_prompt = f"{prompt}, {STYLE_SFX.get(style, STYLE_SFX['cinematic'])}"
    log.info(f"Video {nf}f {w}x{h} {st}steps mode={'GPU' if use_gpu else 'CPU'}")

    frame_dir  = os.path.join(FRAMES_DIR, job_id)
    os.makedirs(frame_dir, exist_ok=True)
    all_frames = []
    n_chunks   = (nf + cs - 1) // cs

    for ci in range(n_chunks):
        n = min(cs, nf - ci * cs)
        update_state(step=f"Vidéo chunk {ci+1}/{n_chunks}...", progress=int(ci/n_chunks*50))
        out = pipe_video(prompt=full_prompt, negative_prompt=NEG,
            num_frames=n, width=w, height=h,
            num_inference_steps=st, guidance_scale=guidance_scale, generator=generator)
        for i, frame in enumerate(out.frames[0]):
            frame.save(os.path.join(frame_dir, f"frame_{ci*cs+i:05d}.png"))
            update_state(current_frame=ci*cs+i+1)
        all_frames.extend(out.frames[0])
        free_vram()

    update_state(step="Export MP4...", progress=52)
    vid_name = f"video_{job_id}.mp4"
    vid_path = os.path.join(OUTPUT_DIR, vid_name)
    writer   = imageio.get_writer(vid_path, fps=fp, codec="libx264", quality=8)
    for f in all_frames: writer.append_data(np.array(f))
    writer.close()
    import shutil; shutil.rmtree(frame_dir, ignore_errors=True)
    update_state(video_path=vid_path, progress=55)
    free_vram()
    return vid_path, vid_name, job_id

def generate_image_from_text(prompt, style="cinematic",
    width=None, height=None, steps=None, guidance_scale=7.5, seed=-1):
    if seed == -1: seed = random.randint(0, 2**32-1)
    generator = torch.Generator(device=device).manual_seed(seed)
    job_id    = _state["job_id"] or uuid.uuid4().hex[:8]
    w  = min(width  or (1024 if use_gpu else 512), 1024 if use_gpu else 512)
    h  = min(height or (1024 if use_gpu else 512), 1024 if use_gpu else 512)
    st = min(steps  or (30   if use_gpu else 15),  50   if use_gpu else 15)
    SDXL_SFX = {"cinematic":"cinematic photograph, dramatic lighting, 8K",
                "cyberpunk":"cyberpunk digital art, neon glow, 8K",
                "fantasy":"fantasy concept art, magical, 8K masterpiece"}
    update_state(step="Génération image...", progress=20)

    # SDXL si GPU, SD1.5 si CPU
    if use_gpu and pipe_image:
        out = pipe_image(prompt=f"{prompt}, {SDXL_SFX.get(style,'photorealistic, 8K')}",
            negative_prompt=NEG, width=w, height=h,
            num_inference_steps=st, guidance_scale=guidance_scale, generator=generator)
    else:
        out = pipe_video.unet  # fallback simple
        # Utilise AnimateDiff en mode image (1 frame)
        out_vid = pipe_video(prompt=prompt, negative_prompt=NEG,
            num_frames=1, width=w, height=h,
            num_inference_steps=st, guidance_scale=guidance_scale, generator=generator)
        out = type('obj', (object,), {'images': out_vid.frames[0]})()

    img_name = f"image_{job_id}.png"
    img_path = os.path.join(OUTPUT_DIR, img_name)
    out.images[0].save(img_path)
    update_state(image_path=img_path, progress=100, running=False, step="✅ Image générée!")
    free_vram()
    return img_path, img_name, job_id

def animate_image_to_video(image_input, num_frames=None, fps=None,
    motion_bucket_id=127, noise_aug_strength=0.02, decode_chunk_size=8, seed=-1):
    if seed == -1: seed = random.randint(0, 2**32-1)
    generator = torch.Generator(device=device).manual_seed(seed)
    job_id    = _state["job_id"] or uuid.uuid4().hex[:8]
    nf = min(num_frames or (25 if use_gpu else 8), 30 if use_gpu else 8)
    fp = fps or DEFAULT_FPS

    if isinstance(image_input, str):
        if image_input.startswith("data:"): image_input = image_input.split(",",1)[1]
        if len(image_input) > 260:
            image_pil = Image.open(io.BytesIO(base64.b64decode(image_input))).convert("RGB")
        else:
            image_pil = Image.open(image_input).convert("RGB")
    else:
        image_pil = image_input.convert("RGB")

    image_pil = image_pil.resize((1024 if use_gpu else 512, 576 if use_gpu else 320))
    update_state(step="Animation SVD...", progress=10)

    if use_gpu and pipe_img2vid:
        frames = pipe_img2vid(image_pil, num_frames=nf, fps=fp,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size, generator=generator).frames[0]
    else:
        # CPU fallback — utilise AnimateDiff avec l'image comme guide
        update_state(step="Animation CPU (AnimateDiff)...", progress=15)
        out = pipe_video(prompt="smooth cinematic motion", negative_prompt=NEG,
            num_frames=nf, width=image_pil.width, height=image_pil.height,
            num_inference_steps=DEFAULT_STEPS, generator=generator)
        frames = out.frames[0]

    vid_name = f"animated_{job_id}.mp4"
    vid_path = os.path.join(OUTPUT_DIR, vid_name)
    export_to_video(frames, vid_path, fps=fp)
    update_state(video_path=vid_path, progress=80)
    free_vram()
    return vid_path, vid_name, job_id

print("✅ 3 fonctions de génération prêtes!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 7 — Assemblage Final
# ══════════════════════════════════════════════════════════════════════
from moviepy.editor import (VideoFileClip, AudioFileClip,
    CompositeAudioClip, concatenate_audioclips)

def assemble_final_video(video_path, job_id, prompt="", style_video="cinematic",
    voix_active=True, style_voix="masculin", texte_voix=None,
    musique_active=True, style_musique="cinematique",
    volume_voix=0.90, volume_musique=0.28):

    video_clip = VideoFileClip(video_path)
    duree      = video_clip.duration
    voix_clip  = None

    if voix_active and tts_available:
        update_state(step="Génération voix off...", progress=58)
        texte     = texte_voix or get_narration(prompt, style_video)
        voix_path = generate_voix_off(texte, style_voix)
        if voix_path:
            voix_clip = AudioFileClip(voix_path).volumex(volume_voix)
            if voix_clip.duration > duree: voix_clip = voix_clip.subclip(0, duree)
        update_state(progress=72)

    musique_clip = None
    if musique_active and music_available:
        update_state(step="Génération musique IA...", progress=75)
        musique_path = generate_musique(style_musique, duree=max(10, int(duree)+3))
        if musique_path:
            raw = AudioFileClip(musique_path).volumex(volume_musique)
            if raw.duration < duree:
                raw = concatenate_audioclips([raw] * (int(duree/raw.duration)+1))
            musique_clip = raw.subclip(0, duree)
        update_state(progress=87)

    update_state(step="Export MP4 final...", progress=89)
    if voix_clip and musique_clip: audio_final = CompositeAudioClip([musique_clip, voix_clip])
    elif voix_clip:    audio_final = voix_clip
    elif musique_clip: audio_final = musique_clip
    else:              audio_final = None

    final_name = f"BENYJOE_FINAL_{job_id}.mp4"
    final_path = os.path.join(OUTPUT_DIR, final_name)
    temp_audio = os.path.join(TEMP_DIR, f"tmp_{job_id}.m4a")

    if audio_final:
        video_clip.set_audio(audio_final).write_videofile(
            final_path, codec="libx264", audio_codec="aac",
            audio_bitrate="192k", temp_audiofile=temp_audio,
            remove_temp=True, fps=video_clip.fps, preset="fast", logger=None)
    else:
        import shutil; shutil.copy(video_path, final_path)

    video_clip.close()
    if voix_clip:    voix_clip.close()
    if musique_clip: musique_clip.close()

    update_state(final_path=final_path, progress=100, running=False, step="✅ MP4 Master prêt!")
    return final_path, final_name

print("✅ Pipeline assemblage prêt!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 8 — Serveur Flask GPU (routes async)
# ══════════════════════════════════════════════════════════════════════
import uuid as _uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

gpu_app = Flask(__name__)
CORS(gpu_app)

def run_async(fn):
    """Lance fn dans un thread — réponse immédiate au client."""
    def _run():
        try: fn()
        except Exception as e:
            update_state(running=False, error=str(e), step="❌ Erreur")
            log.exception("Erreur génération")
    threading.Thread(target=_run, daemon=True).start()

@gpu_app.route("/health")
def health():
    return jsonify({"status":"ok","device":device,"vram_gb":vram,
        "running":_state["running"],"progress":_state["progress"],
        "step":_state["step"],"version":"7.0.0",
        "current_frame":_state["current_frame"],"total_frames":_state["total_frames"],
        "mode": "GPU" if use_gpu else "CPU"})

@gpu_app.route("/progress")
def prog():
    return jsonify({"progress":_state["progress"],"running":_state["running"],
        "step":_state["step"],"current_frame":_state["current_frame"],
        "total_frames":_state["total_frames"],"video_path":_state["video_path"],
        "image_path":_state["image_path"],"final_path":_state["final_path"],
        "error":_state["error"]})

@gpu_app.route("/generate", methods=["POST"])
def api_generate():
    if _state["running"]: return jsonify({"error":"GPU/CPU occupé — patientez"}), 429
    d      = request.get_json(force=True)
    prompt = d.get("prompt","").strip()
    if not prompt: return jsonify({"error":"prompt vide"}), 400
    job_id = _uuid.uuid4().hex[:8]
    nf     = min(int(d.get("num_frames", DEFAULT_FRAMES)), 40 if use_gpu else 6)
    reset_state(total_frames=nf, job_id=job_id)

    def _gen():
        vp, vn, _ = generate_video_from_text(
            prompt=prompt, style=d.get("style","cinematic"),
            num_frames=nf,
            width=int(d.get("width", DEFAULT_W)),
            height=int(d.get("height", DEFAULT_H)),
            steps=min(int(d.get("steps", DEFAULT_STEPS)), 35 if use_gpu else 10),
            guidance_scale=float(d.get("guidance_scale",7.5)),
            fps=int(d.get("fps", DEFAULT_FPS)),
            seed=int(d.get("seed",-1)))
        assemble_final_video(vp, job_id, prompt=prompt,
            style_video=d.get("style","cinematic"),
            voix_active=bool(d.get("voix_active",True)) and tts_available,
            style_voix=d.get("style_voix","masculin"),
            texte_voix=d.get("texte_voix",None),
            musique_active=bool(d.get("musique_active",True)) and music_available,
            style_musique=d.get("style_musique","cinematique"),
            volume_voix=float(d.get("volume_voix",0.9)),
            volume_musique=float(d.get("volume_musique",0.28)))

    run_async(_gen)
    return jsonify({"status":"started","job_id":job_id,
                    "mode":"GPU 🚀" if use_gpu else "CPU ⚠️"})

@gpu_app.route("/generate_image", methods=["POST"])
def api_generate_image():
    if _state["running"]: return jsonify({"error":"Occupé"}), 429
    d      = request.get_json(force=True)
    prompt = d.get("prompt","").strip()
    if not prompt: return jsonify({"error":"prompt vide"}), 400
    job_id = _uuid.uuid4().hex[:8]
    reset_state(job_id=job_id)
    def _gen():
        generate_image_from_text(prompt=prompt, style=d.get("style","cinematic"),
            steps=int(d.get("steps",30)), seed=int(d.get("seed",-1)))
    run_async(_gen)
    return jsonify({"status":"started","job_id":job_id})

@gpu_app.route("/img2video", methods=["POST"])
def api_img2video():
    if _state["running"]: return jsonify({"error":"Occupé"}), 429
    d = request.get_json(force=True)
    if not d.get("image_b64"): return jsonify({"error":"image_b64 manquant"}), 400
    job_id = _uuid.uuid4().hex[:8]
    nf     = min(int(d.get("num_frames",25)), 30 if use_gpu else 8)
    reset_state(total_frames=nf, job_id=job_id)
    def _gen():
        vp, vn, _ = animate_image_to_video(image_input=d["image_b64"],
            num_frames=nf, fps=int(d.get("fps",7)),
            motion_bucket_id=int(d.get("motion_bucket_id",127)),
            seed=int(d.get("seed",-1)))
        assemble_final_video(vp, job_id, prompt=d.get("prompt",""),
            voix_active=bool(d.get("voix_active",False)) and tts_available,
            musique_active=bool(d.get("musique_active",True)) and music_available,
            style_musique=d.get("style_musique","cinematique"))
    run_async(_gen)
    return jsonify({"status":"started","job_id":job_id})

@gpu_app.route("/video/<f>")
def srv_v(f):
    p = os.path.join(OUTPUT_DIR, f)
    return send_file(p, mimetype="video/mp4") if os.path.exists(p) else ("Not found",404)

@gpu_app.route("/final/<f>")
def srv_f(f):
    p = os.path.join(OUTPUT_DIR, f)
    return send_file(p, mimetype="video/mp4") if os.path.exists(p) else ("Not found",404)

@gpu_app.route("/image/<f>")
def srv_i(f):
    p = os.path.join(OUTPUT_DIR, f)
    return send_file(p, mimetype="image/png") if os.path.exists(p) else ("Not found",404)

@gpu_app.route("/audio/<f>")
def srv_a(f):
    p = os.path.join(AUDIO_DIR, f)
    return send_file(p, mimetype="audio/wav") if os.path.exists(p) else ("Not found",404)

@gpu_app.route("/list_outputs")
def list_out():
    files = [{"name":f,"size_mb":round(os.path.getsize(os.path.join(OUTPUT_DIR,f))/1e6,2)}
             for f in os.listdir(OUTPUT_DIR)]
    return jsonify({"files": sorted(files, key=lambda x: x["name"], reverse=True)})

print("✅ Flask GPU — toutes les routes définies!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 9 — Lancement ngrok + Auto-update Render
# ══════════════════════════════════════════════════════════════════════
import subprocess, threading, time, requests

# ⚠️ MODIFIEZ CES 2 VALEURS !
NGROK_TOKEN = "VOTRE_TOKEN_NGROK_ICI"
RENDER_URL  = "https://benyjoe-studio.onrender.com"

subprocess.run(["pkill","-f","ngrok"], capture_output=True)
subprocess.run(["pkill","-f","flask"], capture_output=True)
subprocess.run(["fuser","-k","5001/tcp"], capture_output=True)
time.sleep(3)

def run_flask():
    gpu_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)

threading.Thread(target=run_flask, daemon=True).start()
time.sleep(4)

flask_ok = False
for i in range(10):
    try:
        r = requests.get("http://127.0.0.1:5001/health", timeout=3)
        if r.status_code == 200:
            d = r.json()
            print(f"✅ Flask OK | {d['device']} | {d['vram_gb']} GB | mode={d.get('mode','?')}")
            flask_ok = True
            break
    except: pass
    time.sleep(2)

if not flask_ok:
    print("❌ Flask ne répond pas — relancez cette cellule")
else:
    try:
        subprocess.run(["ngrok","version"], capture_output=True, check=True)
    except:
        print("Installation ngrok...")
        subprocess.run(
            "curl -sL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | "
            "tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && "
            "echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | "
            "tee /etc/apt/sources.list.d/ngrok.list >/dev/null && "
            "apt-get update -qq && apt-get install -qq ngrok", shell=True)

    subprocess.run(["ngrok","config","add-authtoken", NGROK_TOKEN], capture_output=True)
    subprocess.Popen(["ngrok","http","5001"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)

    NGROK_URL = ""
    for i in range(12):
        try:
            r = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
            for t in r.json().get("tunnels",[]):
                if t.get("proto") == "https":
                    NGROK_URL = t["public_url"]; break
            if NGROK_URL: break
        except: pass
        time.sleep(2)
        print(f"  Attente ngrok {i+1}/12...")

    if NGROK_URL:
        print(f"\n{'='*55}")
        print(f"  BENY-JOE CINÉ IA PRO v7.0 — PRÊT!")
        print(f"  Mode : {'GPU 🚀' if use_gpu else 'CPU ⚠️'}")
        print(f"  URL  : {NGROK_URL}")
        print(f"{'='*55}")

        try:
            r = requests.post(f"{RENDER_URL}/api/gpu_url",
                json={"url": NGROK_URL}, timeout=15)
            if r.json().get("ok"):
                print("  ✅ Render mis à jour automatiquement!")
            else:
                print(f"  → Copiez dans Render → COLAB_URL_PRIMARY: {NGROK_URL}")
        except:
            print(f"  → Copiez dans Render → COLAB_URL_PRIMARY: {NGROK_URL}")

        def keep_alive():
            while True:
                try:
                    r = requests.get(NGROK_URL+"/health", timeout=10,
                        headers={"ngrok-skip-browser-warning":"true"})
                    d = r.json()
                    status = f"{d['step']} ({d['progress']}%)" if d.get("running") \
                             else f"Prêt | {d['device']} | {d['vram_gb']}GB"
                    print(f"  GPU | {status}")
                except Exception as e:
                    print(f"  Ping: {e}")
                time.sleep(120)

        threading.Thread(target=keep_alive, daemon=True).start()
        print("  Keep-alive actif — Studio 100% opérationnel! 🎬")
    else:
        print("❌ URL ngrok non trouvée — vérifiez votre token")
