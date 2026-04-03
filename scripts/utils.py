"""
BENY-JOE CINÉ IA PRO v5.0 — Scripts utilitaires
Outils en ligne de commande pour interagir avec le studio
Usage: python scripts/utils.py --help
"""

import argparse, requests, json, base64, os, time, sys
from datetime import datetime

BASE_URL = os.getenv("BENYJOE_URL", "http://localhost:5000")
PIN      = os.getenv("BENYJOE_PIN", "2022002")
HEADERS  = {"Content-Type": "application/json"}


def auth():
    """Vérifie l'authentification."""
    r = requests.post(f"{BASE_URL}/api/auth", json={"pin": PIN}, timeout=10)
    return r.json().get("ok", False)


def wait_for_job(job_id, verbose=True):
    """Attend la fin d'un job et retourne l'URL résultat."""
    while True:
        try:
            r = requests.get(f"{BASE_URL}/api/status/{job_id}", timeout=10)
            d = r.json()
            status = d["status"]
            if verbose:
                pct  = d.get("progress", 0)
                step = d.get("step", "")
                print(f"\r  [{pct:3d}%] {step:<50}", end="", flush=True)
            if status == "done":
                if verbose: print(f"\n✅ Terminé!")
                return d["result"]
            elif status in ("error", "cancelled"):
                if verbose: print(f"\n❌ {status}: {d.get('error','?')}")
                return None
            time.sleep(4)
        except KeyboardInterrupt:
            print("\n⚠️  Interrompu")
            return None


def generate_video(prompt, style="cinematic", frames=24, fps=8, steps=25,
                   voix=True, style_voix="masculin",
                   musique=True, style_musique="cinematique"):
    """Génère une vidéo et attend le résultat."""
    print(f"🎬 Génération vidéo: {prompt[:60]}...")
    payload = {
        "prompt": prompt, "style": style,
        "frames": frames, "fps": fps, "steps": steps,
        "voix_active": voix, "style_voix": style_voix,
        "musique_active": musique, "style_musique": style_musique,
    }
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=15)
    if r.status_code != 200:
        print(f"❌ Erreur: {r.text}")
        return None
    job_id = r.json()["job_id"]
    print(f"   Job: {job_id}")
    return wait_for_job(job_id)


def generate_image(prompt, style="cinematic", resolution="1024x1024", steps=30):
    """Génère une image SDXL."""
    print(f"🖼️  Génération image: {prompt[:60]}...")
    payload = {"prompt": prompt, "style": style, "resolution": resolution, "steps": steps}
    r = requests.post(f"{BASE_URL}/api/generate_image", json=payload, timeout=15)
    if r.status_code != 200:
        print(f"❌ Erreur: {r.text}")
        return None
    return wait_for_job(r.json()["job_id"])


def animate_image(image_path, musique=True, style_musique="cinematique",
                  frames=25, motion=127):
    """Anime une image locale."""
    print(f"🎞️  Animation: {image_path}...")
    if not os.path.exists(image_path):
        print(f"❌ Fichier introuvable: {image_path}")
        return None
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "image_b64": b64, "num_frames": frames,
        "motion_bucket_id": motion,
        "musique_active": musique, "style_musique": style_musique,
    }
    r = requests.post(f"{BASE_URL}/api/img2video", json=payload, timeout=15)
    if r.status_code != 200:
        print(f"❌ Erreur: {r.text}")
        return None
    return wait_for_job(r.json()["job_id"])


def update_gpu_url(url):
    """Met à jour l'URL ngrok Kaggle."""
    r = requests.post(f"{BASE_URL}/api/gpu_url", json={"url": url}, timeout=10)
    d = r.json()
    if d.get("ok"):
        print(f"✅ URL GPU mise à jour: {url}")
    else:
        print(f"❌ Erreur: {d.get('error','?')}")


def health():
    """Affiche l'état du système."""
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    d = r.json()
    print(f"{'='*50}")
    print(f"  BENY-JOE Studio v{d['version']}")
    print(f"  Status: {d['status']}")
    print(f"  Queue:  {d['queue_size']} | Total jobs: {d['jobs_total']} | Done: {d['jobs_done']}")
    for name, info in d.get("gpu", {}).items():
        if isinstance(info, dict):
            s = info.get("status","?")
            v = info.get("vram_gb","?")
            p = info.get("progress",0)
            print(f"  GPU [{name}]: {s} | VRAM: {v} GB | Progress: {p}%")
        else:
            print(f"  GPU [{name}]: {info}")
    print(f"{'='*50}")


def history():
    """Affiche l'historique des jobs."""
    r = requests.get(f"{BASE_URL}/api/history", timeout=10)
    jobs = r.json().get("jobs", [])
    print(f"\n{'='*60}")
    print(f"  Historique — {len(jobs)} jobs")
    print(f"{'='*60}")
    for j in jobs[:20]:
        status = j["status"]
        prompt = (j.get("prompt","")[:40] + "...") if len(j.get("prompt","")) > 40 else j.get("prompt","")
        created = j.get("created_at","")[:16]
        print(f"  [{status:12s}] {created} | {prompt}")
    print(f"{'='*60}\n")


def clear_queue():
    """Vide la file d'attente."""
    r = requests.post(f"{BASE_URL}/api/clear_queue", timeout=10)
    d = r.json()
    print(f"✅ {d.get('cleared',0)} jobs supprimés de la file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BENY-JOE Studio v5.0 — Utilitaires CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python scripts/utils.py health
  python scripts/utils.py video "Un samouraï sous la pluie" --style cinematic
  python scripts/utils.py image "Portait guerrier, armure dorée" --style fantasy
  python scripts/utils.py animate photo.jpg --motion 180
  python scripts/utils.py gpu-url https://xxxx.ngrok-free.app
  python scripts/utils.py history
  python scripts/utils.py clear
        """
    )
    parser.add_argument("--url", default=BASE_URL, help="URL du serveur")
    parser.add_argument("--pin", default=PIN, help="Code PIN")
    sub = parser.add_subparsers(dest="cmd")

    # Health
    sub.add_parser("health", help="État du système")

    # History
    sub.add_parser("history", help="Historique des jobs")

    # Clear
    sub.add_parser("clear", help="Vider la file")

    # Video
    p_vid = sub.add_parser("video", help="Générer une vidéo")
    p_vid.add_argument("prompt")
    p_vid.add_argument("--style",  default="cinematic")
    p_vid.add_argument("--frames", type=int, default=24)
    p_vid.add_argument("--fps",    type=int, default=8)
    p_vid.add_argument("--steps",  type=int, default=25)
    p_vid.add_argument("--voix",   default="masculin")
    p_vid.add_argument("--musique",default="cinematique")
    p_vid.add_argument("--no-voix", action="store_true")
    p_vid.add_argument("--no-musique", action="store_true")

    # Image
    p_img = sub.add_parser("image", help="Générer une image")
    p_img.add_argument("prompt")
    p_img.add_argument("--style",      default="cinematic")
    p_img.add_argument("--resolution", default="1024x1024")
    p_img.add_argument("--steps",      type=int, default=30)

    # Animate
    p_ani = sub.add_parser("animate", help="Animer une image")
    p_ani.add_argument("image", help="Chemin vers l'image")
    p_ani.add_argument("--frames",  type=int, default=25)
    p_ani.add_argument("--motion",  type=int, default=127)
    p_ani.add_argument("--musique", default="cinematique")
    p_ani.add_argument("--no-musique", action="store_true")

    # GPU URL
    p_gpu = sub.add_parser("gpu-url", help="Mettre à jour l'URL GPU")
    p_gpu.add_argument("url", help="Nouvelle URL ngrok")

    args = parser.parse_args()
    if args.url: BASE_URL = args.url
    if args.pin: PIN = args.pin

    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    if args.cmd == "health":
        health()
    elif args.cmd == "history":
        history()
    elif args.cmd == "clear":
        clear_queue()
    elif args.cmd == "video":
        url = generate_video(
            args.prompt, args.style, args.frames, args.fps, args.steps,
            not args.no_voix, args.voix,
            not args.no_musique, args.musique,
        )
        if url: print(f"🎬 Résultat: {url}")
    elif args.cmd == "image":
        url = generate_image(args.prompt, args.style, args.resolution, args.steps)
        if url: print(f"🖼️  Résultat: {url}")
    elif args.cmd == "animate":
        url = animate_image(args.image, not args.no_musique, args.musique, args.frames, args.motion)
        if url: print(f"🎞️  Résultat: {url}")
    elif args.cmd == "gpu-url":
        update_gpu_url(args.url)
