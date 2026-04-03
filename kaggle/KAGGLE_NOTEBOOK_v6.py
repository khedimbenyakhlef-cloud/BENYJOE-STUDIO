# ╔══════════════════════════════════════════════════════════════════════╗
# ║   BENY-JOE CINÉ IA PRO v6.0 — NOTEBOOK COLAB / KAGGLE              ║
# ║   Fondé par KHEDIM BENYAKHLEF dit BENY-JOE                          ║
# ║                                                                      ║
# ║   INSTRUCTIONS :                                                     ║
# ║   KAGGLE  → Settings → Accelerator → GPU T4 x2 + Internet ON        ║
# ║   COLAB   → Exécution → Modifier le type d'exécution → GPU T4        ║
# ║   Copier chaque bloc dans UNE cellule séparée                        ║
# ║   Run All → copier l'URL affichée → coller dans Render               ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════
# CELLULE 1 — Installation complète
# ══════════════════════════════════════════════════════════════════════
import subprocess, sys

def pip(pkg):
    r = subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True, text=True)
    status = "OK" if r.returncode == 0 else f"WARN: {r.stderr[-80:]}"
    print(f"  {pkg[:45]:<45} {status}")

print("=== Installation BENY-JOE STUDIO v6.0 ===")
print("\n[1/6] Core diffusion...")
pip("diffusers==0.25.0")
pip("transformers==4.38.2")
pip("accelerate==0.27.2")
pip("omegaconf einops safetensors")

print("\n[2/6] Image / Vidéo...")
pip("imageio==2.33.1 imageio-ffmpeg")
pip("Pillow>=10.0.0 opencv-python-headless")

print("\n[3/6] Voix Off (Coqui TTS)...")
pip("TTS==0.22.0")

print("\n[4/6] Musique IA (AudioCraft)...")
pip("git+https://github.com/facebookresearch/audiocraft.git")

print("\n[5/6] Assemblage audio/vidéo...")
pip("moviepy==1.0.3 ffmpeg-python pydub scipy")

print("\n[6/6] Serveur Flask...")
pip("flask==3.0.3 flask-cors==4.0.1 requests")

print("\n✅ Installation terminée!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 2 — Init GPU + état global + dossiers
# ══════════════════════════════════════════════════════════════════════
import torch, os, uuid, random, threading, time, logging
import base64, io, numpy as np, imageio
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram     = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"GPU: {gpu_name} | VRAM: {vram} GB | CUDA: {torch.version.cuda}")
else:
    vram = 0
    print("CPU mode")

# Dossiers — compatibles Colab ET Kaggle
if os.path.exists("/kaggle/working"):
    BASE_WORK = "/kaggle/working"
else:
    BASE_WORK = "/content"

OUTPUT_DIR = os.path.join(BASE_WORK, "outputs")
FRAMES_DIR = os.path.join(BASE_WORK, "frames")
AUDIO_DIR  = os.path.join(BASE_WORK, "audio")
TEMP_DIR   = os.path.join(BASE_WORK, "temp")
for d in [OUTPUT_DIR, FRAMES_DIR, AUDIO_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("BenyJoe")

# État global thread-safe
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

NGROK_URL = ""  # sera défini cellule 11
print(f"Init OK | device={device} | vram={vram} GB")
print(f"Dossiers: {OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 3 — AnimateDiff (Texte → Vidéo)
# ══════════════════════════════════════════════════════════════════════
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_video

print("Chargement AnimateDiff... (~2 GB, patience 3-5 min)")

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16,
)
pipe_video = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16,
)
pipe_video.scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe_video = pipe_video.to(device)
pipe_video.vae.enable_slicing()
pipe_video.enable_attention_slicing()
try: pipe_video.enable_vae_tiling()
except Exception: pass

print("AnimateDiff prêt!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 4 — SDXL (Texte → Image 1024px)
# ══════════════════════════════════════════════════════════════════════
from diffusers import StableDiffusionXLPipeline

print("Chargement SDXL... (~7 GB, patience 5-10 min)")

pipe_image = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe_image = pipe_image.to(device)
pipe_image.enable_attention_slicing()
try: pipe_image.enable_vae_tiling()
except Exception: pass

print("SDXL prêt!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 5 — SVD (Image → Vidéo animée)
# ══════════════════════════════════════════════════════════════════════
from diffusers import StableVideoDiffusionPipeline

print("Chargement SVD... (~8 GB, patience 5-10 min)")

pipe_img2vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe_img2vid = pipe_img2vid.to(device)
pipe_img2vid.enable_attention_slicing()
try: pipe_img2vid.enable_vae_tiling()
except Exception: pass

print("SVD prêt!")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 6 — Coqui XTTS (Voix Off)
# ══════════════════════════════════════════════════════════════════════
from TTS.api import TTS as CoquiTTS

print("Chargement Coqui XTTS v2... (~2 GB)")
os.environ["COQUI_TOS_AGREED"] = "1"

tts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

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
    "scifi":       "Aux confins de l'univers, une nouvelle découverte attend l'humanité.",
    "noir":        "Dans les ruelles sombres, les secrets ne restent jamais enterrés.",
    "fantasy":     "Dans ce monde de magie et de merveilles, tout est possible.",
    "horror":      "L'obscurité cache des secrets que l'homme n'est pas fait pour connaître.",
    "romantique":  "Dans la lumière dorée de cet instant, l'amour trouve sa plus belle forme.",
}

def generate_voix_off(texte, style_voix="masculin", output_path=None):
    if not output_path:
        output_path = os.path.join(AUDIO_DIR, f"voix_{uuid.uuid4().hex[:8]}.wav")
    p = VOICE_PROFILES.get(style_voix, VOICE_PROFILES["masculin"])
    log.info(f"Voix [{style_voix}] {texte[:60]}")
    tts_model.tts_to_file(
        text=texte, speaker=p["speaker"],
        language=p["language"], file_path=output_path, speed=p["speed"],
    )
    return output_path

def get_narration(prompt, style):
    base  = NARRATION.get(style, "Une vision cinématographique unique.")
    extra = (prompt[:100] + ".") if prompt else ""
    return f"{base} {extra}".strip()

print(f"Coqui XTTS prêt | Voix: {list(VOICE_PROFILES.keys())}")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 7 — MusicGen (Musique IA)
# ══════════════════════════════════════════════════════════════════════
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

print("Chargement MusicGen small... (~1 GB)")

music_model = MusicGen.get_pretrained("facebook/musicgen-small")
music_model.set_generation_params(duration=10)

MUSIC_PROMPTS = {
    "cinematique": "epic cinematic orchestral, dramatic strings, powerful brass, Hans Zimmer style",
    "ambiante":    "peaceful ambient, soft piano, atmospheric pads, calm, Brian Eno style",
    "cyberpunk":   "dark synthwave, futuristic industrial beats, neon atmosphere, driving bass",
    "nature":      "acoustic guitar, birds, gentle rain, folk instrumental, peaceful",
    "scifi":       "space ambient, ethereal pads, cosmic, futuristic synthesizers, mysterious",
    "noir":        "jazz saxophone, trumpet, rainy night, 1940s detective, moody",
    "fantasy":     "magical orchestral, choir, enchanted forest, mystical harps, epic",
    "horror":      "dissonant strings, tension, suspense, eerie, terrifying atmosphere",
    "epic_battle": "powerful timpani drums, war horns, intense strings, triumphant brass",
    "romantique":  "romantic violin solo, gentle piano, emotional and tender, love theme",
}

def generate_musique(style="cinematique", duree=10, output_path=None):
    if not output_path:
        output_path = os.path.join(AUDIO_DIR, f"musique_{uuid.uuid4().hex[:8]}")
    prompt = MUSIC_PROMPTS.get(style, MUSIC_PROMPTS["cinematique"])
    log.info(f"Musique [{style}] {duree}s")
    music_model.set_generation_params(duration=max(5, int(duree)))
    wav = music_model.generate([prompt])
    audio_write(output_path, wav[0].cpu(), music_model.sample_rate,
                strategy="loudness", loudness_compressor=True)
    return output_path + ".wav"

print(f"MusicGen prêt | Styles: {list(MUSIC_PROMPTS.keys())}")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 8 — Fonctions de génération
# ══════════════════════════════════════════════════════════════════════
NEG = (
    "blurry, low quality, bad anatomy, deformed, ugly, duplicate, "
    "error, jpeg artifacts, watermark, text, cartoon, flat colors"
)
STYLE_SFX = {
    "cinematic":   "cinematic lighting, dramatic shadows, anamorphic lens, film grain, 4K HDR",
    "cyberpunk":   "neon lights, rain-soaked streets, volumetric fog, ultra realistic, 8K",
    "epic_battle": "fire and smoke, dramatic sky, hyperrealistic warriors, 8K",
    "nature":      "golden hour, misty mountains, ultra wide angle, cinematic grade, 4K",
    "scifi":       "space station, zero gravity, distant nebula, sci-fi tech, cinematic",
    "noir":        "black and white film noir, rain, smoky atmosphere, dramatic shadows, 1940s",
    "fantasy":     "enchanted forest, magical creatures, bioluminescent, epic fantasy, 8K",
    "horror":      "fog, moonlight, horror atmosphere, dark, cinematic, ultra realistic",
    "romantique":  "golden hour, warm bokeh, emotional, dreamy, soft light, romantic, 4K",
}

# A — Texte → Vidéo AnimateDiff
def generate_video_from_text(
    prompt, style="cinematic",
    num_frames=24, width=512, height=512,
    steps=25, guidance_scale=7.5, fps=8, seed=-1, chunk_size=8,
):
    if seed == -1: seed = random.randint(0, 2**32 - 1)
    generator    = torch.Generator(device=device).manual_seed(seed)
    job_id       = _state["job_id"] or uuid.uuid4().hex[:8]
    full_prompt  = f"{prompt}, {STYLE_SFX.get(style, STYLE_SFX['cinematic'])}"
    log.info(f"Video {num_frames}f {width}x{height} {steps}steps seed={seed}")

    frame_dir  = os.path.join(FRAMES_DIR, job_id)
    os.makedirs(frame_dir, exist_ok=True)
    all_frames = []
    n_chunks   = (num_frames + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        n = min(chunk_size, num_frames - ci * chunk_size)
        update_state(
            step=f"Vidéo chunk {ci+1}/{n_chunks}...",
            progress=int(ci / n_chunks * 50),
        )
        out = pipe_video(
            prompt=full_prompt, negative_prompt=NEG,
            num_frames=n, width=width, height=height,
            num_inference_steps=steps, guidance_scale=guidance_scale,
            generator=generator,
        )
        for i, frame in enumerate(out.frames[0]):
            idx = ci * chunk_size + i
            frame.save(os.path.join(frame_dir, f"frame_{idx:05d}.png"))
            update_state(current_frame=idx + 1)
        all_frames.extend(out.frames[0])
        free_vram()

    update_state(step="Export MP4...", progress=52)
    vid_name = f"video_{job_id}.mp4"
    vid_path = os.path.join(OUTPUT_DIR, vid_name)
    writer   = imageio.get_writer(vid_path, fps=fps, codec="libx264", quality=8)
    for f in all_frames:
        writer.append_data(np.array(f))
    writer.close()

    import shutil; shutil.rmtree(frame_dir, ignore_errors=True)
    update_state(video_path=vid_path, progress=55)
    free_vram()
    log.info(f"Video OK: {vid_path}")
    return vid_path, vid_name, job_id

# B — Texte → Image SDXL
def generate_image_from_text(
    prompt, style="cinematic",
    width=1024, height=1024,
    steps=30, guidance_scale=7.5, seed=-1,
):
    if seed == -1: seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    job_id    = _state["job_id"] or uuid.uuid4().hex[:8]
    SDXL_SFX  = {
        "cinematic":   "cinematic photograph, dramatic lighting, 8K, photorealistic",
        "cyberpunk":   "cyberpunk digital art, neon glow, ultra detailed, 8K",
        "noir":        "noir photography, high contrast, film grain, 1940s",
        "fantasy":     "fantasy concept art, magical, ethereal glow, 8K, masterpiece",
        "scifi":       "sci-fi concept art, futuristic, ultra detailed, 8K",
        "horror":      "horror art, dark, atmospheric, ultra realistic, 8K",
        "nature":      "nature photography, golden hour, 8K, award winning",
        "epic_battle": "epic digital art, dramatic scene, 8K, masterpiece",
        "romantique":  "romantic photography, warm bokeh, soft light, 4K",
    }
    full_prompt = f"{prompt}, {SDXL_SFX.get(style, SDXL_SFX['cinematic'])}"
    update_state(step="Génération SDXL...", progress=20)

    out      = pipe_image(
        prompt=full_prompt, negative_prompt=NEG,
        width=width, height=height,
        num_inference_steps=steps, guidance_scale=guidance_scale,
        generator=generator,
    )
    img_name = f"image_{job_id}.png"
    img_path = os.path.join(OUTPUT_DIR, img_name)
    out.images[0].save(img_path)

    update_state(image_path=img_path, progress=100, running=False, step="Image générée!")
    free_vram()
    log.info(f"Image OK: {img_path}")
    return img_path, img_name, job_id

# C — Image → Vidéo SVD
def animate_image_to_video(
    image_input, num_frames=25, fps=7,
    motion_bucket_id=127, noise_aug_strength=0.02,
    decode_chunk_size=8, seed=-1,
):
    if seed == -1: seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    job_id    = _state["job_id"] or uuid.uuid4().hex[:8]

    if isinstance(image_input, str):
        if image_input.startswith("data:"):
            image_input = image_input.split(",", 1)[1]
        if len(image_input) > 260:
            img_bytes = base64.b64decode(image_input)
            image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            image_pil = Image.open(image_input).convert("RGB")
    else:
        image_pil = image_input.convert("RGB")

    image_pil = image_pil.resize((1024, 576))
    update_state(step="Animation SVD...", progress=10)

    frames = pipe_img2vid(
        image_pil, num_frames=num_frames, fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        decode_chunk_size=decode_chunk_size,
        generator=generator,
    ).frames[0]

    vid_name = f"animated_{job_id}.mp4"
    vid_path = os.path.join(OUTPUT_DIR, vid_name)
    export_to_video(frames, vid_path, fps=fps)

    update_state(video_path=vid_path, progress=80)
    free_vram()
    log.info(f"Animation OK: {vid_path}")
    return vid_path, vid_name, job_id

print("3 fonctions de génération prêtes ✅")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 9 — Assemblage Final (Vidéo + Voix + Musique)
# ══════════════════════════════════════════════════════════════════════
from moviepy.editor import (
    VideoFileClip, AudioFileClip,
    CompositeAudioClip, concatenate_audioclips,
)

def assemble_final_video(
    video_path, job_id,
    prompt="", style_video="cinematic",
    voix_active=True,    style_voix="masculin",    texte_voix=None,
    musique_active=True, style_musique="cinematique",
    volume_voix=0.90,    volume_musique=0.28,
):
    log.info(f"Assemblage | voix={voix_active} | musique={musique_active}")
    video_clip = VideoFileClip(video_path)
    duree      = video_clip.duration

    voix_clip = None
    if voix_active:
        update_state(step="Génération voix off...", progress=58)
        texte     = texte_voix or get_narration(prompt, style_video)
        voix_path = generate_voix_off(texte, style_voix)
        voix_clip = AudioFileClip(voix_path).volumex(volume_voix)
        if voix_clip.duration > duree:
            voix_clip = voix_clip.subclip(0, duree)
        update_state(progress=72)

    musique_clip = None
    if musique_active:
        update_state(step="Génération musique IA...", progress=75)
        musique_path = generate_musique(style_musique, duree=max(10, int(duree) + 3))
        raw          = AudioFileClip(musique_path).volumex(volume_musique)
        if raw.duration < duree:
            raw = concatenate_audioclips([raw] * (int(duree / raw.duration) + 1))
        musique_clip = raw.subclip(0, duree)
        update_state(progress=87)

    update_state(step="Mixage audio...", progress=89)
    if voix_clip and musique_clip:
        audio_final = CompositeAudioClip([musique_clip, voix_clip])
    elif voix_clip:
        audio_final = voix_clip
    elif musique_clip:
        audio_final = musique_clip
    else:
        audio_final = None

    update_state(step="Export MP4 final...", progress=92)
    final_name = f"BENYJOE_FINAL_{job_id}.mp4"
    final_path = os.path.join(OUTPUT_DIR, final_name)
    temp_audio = os.path.join(TEMP_DIR, f"tmp_{job_id}.m4a")

    if audio_final:
        video_clip.set_audio(audio_final).write_videofile(
            final_path,
            codec="libx264", audio_codec="aac",
            audio_bitrate="192k",
            temp_audiofile=temp_audio,
            remove_temp=True,
            fps=video_clip.fps,
            preset="fast",
            logger=None,
        )
    else:
        import shutil; shutil.copy(video_path, final_path)

    video_clip.close()
    if voix_clip:    voix_clip.close()
    if musique_clip: musique_clip.close()

    update_state(final_path=final_path, progress=100, running=False, step="✅ MP4 Master prêt!")
    log.info(f"FINAL OK: {final_path}")
    return final_path, final_name

print("Pipeline assemblage prêt ✅")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 10 — Serveur Flask GPU (toutes les routes)
# ══════════════════════════════════════════════════════════════════════
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

gpu_app = Flask(__name__)
CORS(gpu_app)

@gpu_app.route("/health")
def health():
    return jsonify({
        "status": "ok", "device": device, "vram_gb": vram,
        "running": _state["running"], "progress": _state["progress"],
        "step": _state["step"], "version": "6.0.0",
        "current_frame": _state["current_frame"],
        "total_frames":  _state["total_frames"],
    })

@gpu_app.route("/progress")
def prog():
    return jsonify({
        "progress":      _state["progress"],
        "running":       _state["running"],
        "step":          _state["step"],
        "current_frame": _state["current_frame"],
        "total_frames":  _state["total_frames"],
        "video_path":    _state["video_path"],
        "image_path":    _state["image_path"],
        "final_path":    _state["final_path"],
        "error":         _state["error"],
    })

def run_generation_async(fn, *args, **kwargs):
    """Lance une génération dans un thread séparé."""
    def _run():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            update_state(running=False, error=str(e), step="❌ Erreur")
            log.exception("Erreur génération async")
    threading.Thread(target=_run, daemon=True).start()

@gpu_app.route("/generate", methods=["POST"])
def api_generate():
    if _state["running"]:
        return jsonify({"error": "GPU occupé — patientez"}), 429
    d      = request.get_json(force=True)
    prompt = d.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt vide"}), 400

    job_id = uuid.uuid4().hex[:8]
    nf     = min(int(d.get("num_frames", 24)), 40)
    reset_state(total_frames=nf, job_id=job_id)

    def _gen():
        global NGROK_URL
        vp, vn, _ = generate_video_from_text(
            prompt=prompt, style=d.get("style", "cinematic"),
            num_frames=nf,
            width=int(d.get("width", 512)), height=int(d.get("height", 512)),
            steps=min(int(d.get("steps", 25)), 35),
            guidance_scale=float(d.get("guidance_scale", 7.5)),
            fps=int(d.get("fps", 8)), seed=int(d.get("seed", -1)),
        )
        fp, fn_final = assemble_final_video(
            video_path=vp, job_id=job_id, prompt=prompt,
            style_video=d.get("style", "cinematic"),
            voix_active=bool(d.get("voix_active", True)),
            style_voix=d.get("style_voix", "masculin"),
            texte_voix=d.get("texte_voix", None),
            musique_active=bool(d.get("musique_active", True)),
            style_musique=d.get("style_musique", "cinematique"),
            volume_voix=float(d.get("volume_voix", 0.9)),
            volume_musique=float(d.get("volume_musique", 0.28)),
        )
        update_state(
            final_path=fp, progress=100, running=False, step="✅ Terminé!"
        )

    run_generation_async(_gen)
    return jsonify({"status": "started", "job_id": job_id})

@gpu_app.route("/generate_image", methods=["POST"])
def api_generate_image():
    if _state["running"]:
        return jsonify({"error": "GPU occupé"}), 429
    d      = request.get_json(force=True)
    prompt = d.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt vide"}), 400

    res    = d.get("resolution", "1024x1024").split("x")
    job_id = uuid.uuid4().hex[:8]
    reset_state(job_id=job_id)

    def _gen():
        generate_image_from_text(
            prompt=prompt, style=d.get("style", "cinematic"),
            width=int(res[0]) if len(res) == 2 else 1024,
            height=int(res[1]) if len(res) == 2 else 1024,
            steps=min(int(d.get("steps", 30)), 50),
            guidance_scale=float(d.get("guidance_scale", 7.5)),
            seed=int(d.get("seed", -1)),
        )

    run_generation_async(_gen)
    return jsonify({"status": "started", "job_id": job_id})

@gpu_app.route("/img2video", methods=["POST"])
def api_img2video():
    if _state["running"]:
        return jsonify({"error": "GPU occupé"}), 429
    d = request.get_json(force=True)
    if not d.get("image_b64"):
        return jsonify({"error": "image_b64 manquant"}), 400

    job_id = uuid.uuid4().hex[:8]
    nf     = int(d.get("num_frames", 25))
    reset_state(total_frames=nf, job_id=job_id)

    def _gen():
        vp, vn, _ = animate_image_to_video(
            image_input=d["image_b64"],
            num_frames=min(nf, 30),
            fps=int(d.get("fps", 7)),
            motion_bucket_id=int(d.get("motion_bucket_id", 127)),
            seed=int(d.get("seed", -1)),
        )
        fp, fn_final = assemble_final_video(
            video_path=vp, job_id=job_id, prompt=d.get("prompt", ""),
            style_video="cinematic",
            voix_active=bool(d.get("voix_active", False)),
            style_voix=d.get("style_voix", "masculin"),
            texte_voix=d.get("texte_voix", ""),
            musique_active=bool(d.get("musique_active", True)),
            style_musique=d.get("style_musique", "cinematique"),
        )
        update_state(final_path=fp, progress=100, running=False, step="✅ Terminé!")

    run_generation_async(_gen)
    return jsonify({"status": "started", "job_id": job_id})

@gpu_app.route("/generate_voix", methods=["POST"])
def api_voix():
    d = request.get_json(force=True)
    texte = d.get("texte", "").strip()
    if not texte: return jsonify({"error": "texte vide"}), 400
    try:
        path = generate_voix_off(texte, d.get("style_voix", "masculin"))
        return jsonify({"audio_url": NGROK_URL + "/audio/" + os.path.basename(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@gpu_app.route("/generate_musique", methods=["POST"])
def api_musique():
    d = request.get_json(force=True)
    try:
        path = generate_musique(d.get("style", "cinematique"), int(d.get("duree", 10)))
        return jsonify({"audio_url": NGROK_URL + "/audio/" + os.path.basename(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    files = []
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        files.append({"name": f, "size_mb": round(os.path.getsize(fp)/1e6, 2),
                      "url": NGROK_URL + "/final/" + f})
    return jsonify({"files": sorted(files, key=lambda x: x["name"], reverse=True)})

print("Serveur Flask GPU — toutes les routes définies ✅")

# ══════════════════════════════════════════════════════════════════════
# CELLULE 11 — Lancement ngrok + Flask + Keep-alive + Auto-update Render
# ══════════════════════════════════════════════════════════════════════
import subprocess, threading, time, requests

# ─────────────────────────────────────────────────────────────────────
# ⚠️  MODIFIEZ CES 2 VALEURS AVANT DE LANCER !
# ─────────────────────────────────────────────────────────────────────
NGROK_TOKEN  = "VOTRE_TOKEN_NGROK_ICI"       # https://dashboard.ngrok.com/auth
RENDER_URL   = "https://benyjoe-studio.onrender.com"  # URL de votre app Render
# ─────────────────────────────────────────────────────────────────────

# Nettoyage
subprocess.run(["pkill", "-f", "ngrok"],  capture_output=True)
subprocess.run(["pkill", "-f", "flask"],  capture_output=True)
subprocess.run(["fuser", "-k", "5001/tcp"], capture_output=True)
time.sleep(3)

# Lancer Flask en thread daemon
def run_flask():
    gpu_app.run(host="0.0.0.0", port=5001,
                debug=False, use_reloader=False, threaded=True)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(4)

# Vérifier Flask
flask_ok = False
for i in range(10):
    try:
        r = requests.get("http://127.0.0.1:5001/health", timeout=3)
        if r.status_code == 200:
            d = r.json()
            print(f"Flask OK | device={d['device']} | vram={d['vram_gb']} GB")
            flask_ok = True
            break
    except Exception:
        pass
    time.sleep(2)

if not flask_ok:
    print("❌ Flask ne répond pas — relancez cette cellule")
else:
    # Installer ngrok si absent
    try:
        subprocess.run(["ngrok", "version"], capture_output=True, check=True)
    except Exception:
        print("Installation ngrok...")
        subprocess.run(
            "curl -sL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | "
            "tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && "
            "echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | "
            "tee /etc/apt/sources.list.d/ngrok.list >/dev/null && "
            "apt-get update -qq && apt-get install -qq ngrok",
            shell=True
        )

    # Configurer token
    subprocess.run(["ngrok", "config", "add-authtoken", NGROK_TOKEN], capture_output=True)

    # Lancer ngrok
    ngrok_proc = subprocess.Popen(
        ["ngrok", "http", "5001", "--log=stdout", "--log-format=json"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(5)

    # Récupérer l'URL ngrok
    NGROK_URL = ""
    for attempt in range(12):
        try:
            r = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
            for t in r.json().get("tunnels", []):
                if t.get("proto") == "https":
                    NGROK_URL = t["public_url"]
                    break
            if NGROK_URL:
                break
        except Exception:
            pass
        time.sleep(2)
        print(f"  Attente ngrok... {attempt+1}/12")

    if NGROK_URL:
        print("")
        print("=" * 60)
        print("  BENY-JOE CINÉ IA PRO v6.0 — GPU PRÊT!")
        print(f"  URL GPU : {NGROK_URL}")
        print("=" * 60)

        # ── AUTO-UPDATE Render ────────────────────────────────────────
        try:
            r = requests.post(
                f"{RENDER_URL}/api/gpu_url",
                json={"url": NGROK_URL},
                timeout=15,
            )
            if r.json().get("ok"):
                print(f"  ✅ Render mis à jour automatiquement!")
            else:
                print(f"  ⚠️  Render: {r.json()}")
        except Exception as e:
            print(f"  ⚠️  Render auto-update: {e}")
            print(f"  → Copiez manuellement dans Render → COLAB_URL_PRIMARY:")
            print(f"  → {NGROK_URL}")
        print("=" * 60)

        # ── Keep-alive — ping toutes les 2 min ───────────────────────
        def keep_alive():
            while True:
                try:
                    r = requests.get(
                        NGROK_URL + "/health", timeout=10,
                        headers={"ngrok-skip-browser-warning": "true"},
                    )
                    d = r.json()
                    if d.get("running"):
                        print(f"  GPU | {d['step']} ({d['progress']}%) | VRAM: {d['vram_gb']} GB")
                    else:
                        print(f"  GPU Prêt | VRAM: {d['vram_gb']} GB")
                except Exception as e:
                    print(f"  Keep-alive: {e}")
                time.sleep(120)

        threading.Thread(target=keep_alive, daemon=True).start()
        print("\n  Keep-alive actif (ping toutes les 2 min)")
        print("  Studio prêt — Vidéo + Image + Animation + Voix + Musique!")

    else:
        print("❌ URL ngrok non trouvée")
        print("   Vérifiez votre token ngrok et réessayez")
