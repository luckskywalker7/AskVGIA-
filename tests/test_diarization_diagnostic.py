# tests/test_diarization_diagnostic.py
"""
Diagnostic pré-flight — diarisation pyannote
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vérifie AVANT de lancer le pipeline :
  1. Token HuggingFace valide
  2. Accès aux 4 modèles pyannote requis
  3. soundfile installé
  4. FFmpeg disponible
  5. Modèle WhisperX chargeable

Usage :
    python tests/test_diarization_diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.WARNING)   # muet sauf erreurs critiques

from config.settings import DIARIZATION
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────

OK   = "✅"
WARN = "⚠️ "
ERR  = "❌"

REQUIRED_MODELS = [
    ("pyannote/speaker-diarization-3.1",
     "https://huggingface.co/pyannote/speaker-diarization-3.1"),
    ("pyannote/speaker-diarization-community-1",
     "https://huggingface.co/pyannote/speaker-diarization-community-1"),
    ("pyannote/segmentation-3.0",
     "https://huggingface.co/pyannote/segmentation-3.0"),
    ("pyannote/wespeaker-voxceleb-resnet34-LM",
     "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM"),
]


def check_token() -> bool:
    """Vérifie que le token HF est configuré et non vide."""
    token = DIARIZATION.HF_TOKEN
    if not token or token.startswith("hf_VOTRE"):
        print(f"{ERR} Token HF manquant dans config/settings.py")
        print("   → Définissez DIARIZATION.HF_TOKEN = 'hf_xxxxx'")
        return False

    # Validation basique du format
    if not token.startswith("hf_") or len(token) < 20:
        print(f"{WARN} Token HF suspect (format inhabituel) : {token[:10]}…")
        return False

    print(f"{OK} Token HF configuré : {token[:10]}…")
    return True


def check_hf_model_access(model_id: str, token: str) -> bool:
    """Teste l'accès HTTP à un modèle HuggingFace gated."""
    try:
        import requests
        url     = f"https://huggingface.co/api/models/{model_id}"
        headers = {"Authorization": f"Bearer {token}"}
        r       = requests.get(url, headers=headers, timeout=10)

        if r.status_code == 200:
            return True
        elif r.status_code == 401:
            print(f"   {ERR} 401 — Token invalide ou expiré")
            return False
        elif r.status_code == 403:
            print(f"   {ERR} 403 — Modèle gated : conditions non acceptées")
            return False
        elif r.status_code == 404:
            print(f"   {ERR} 404 — Modèle introuvable : {model_id}")
            return False
        else:
            print(f"   {WARN} Statut HTTP inattendu : {r.status_code}")
            return False

    except ImportError:
        print(f"   {WARN} requests absent → vérification ignorée")
        return True   # Optimiste si requests pas dispo
    except Exception as exc:
        print(f"   {WARN} Erreur réseau : {exc}")
        return False


def check_models_access() -> bool:
    """Vérifie l'accès à chacun des 4 modèles requis."""
    token   = DIARIZATION.HF_TOKEN
    all_ok  = True

    print("\n📋 Vérification des accès HuggingFace :")
    print(f"   {'Modèle':<50s} {'Statut':>8}")
    print(f"   {'-'*50}---------")

    for model_id, url in REQUIRED_MODELS:
        ok = check_hf_model_access(model_id, token)
        status = f"{OK} OK" if ok else f"{ERR} REFUSÉ"
        print(f"   {model_id:<50s} {status}")
        if not ok:
            print(f"   → Accepter les conditions : {url}")
            all_ok = False

    return all_ok


def check_soundfile() -> bool:
    """Vérifie que soundfile est installé."""
    try:
        import soundfile as sf
        print(f"{OK} soundfile installé (v{sf.__version__})")
        return True
    except ImportError:
        print(f"{ERR} soundfile absent → pip install soundfile")
        return False


def check_ffmpeg() -> bool:
    """Vérifie que ffmpeg est disponible dans le PATH."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"{OK} FFmpeg disponible : {version_line}")
            return True
        else:
            print(f"{ERR} ffmpeg renvoie une erreur")
            return False
    except FileNotFoundError:
        print(
            f"{ERR} FFmpeg introuvable dans le PATH\n"
            f"   → Windows : winget install Gyan.FFmpeg\n"
            f"   → Redémarrer le terminal après l'installation"
        )
        return False


def check_pyannote() -> bool:
    """Vérifie que pyannote.audio est importable."""
    try:
        import pyannote.audio
        print(f"{OK} pyannote.audio importable (v{pyannote.audio.__version__})")
        return True
    except ImportError:
        print(f"{ERR} pyannote.audio absent → pip install pyannote.audio")
        return False


def check_whisperx() -> bool:
    """Vérifie que whisperx est importable."""
    try:
        import whisperx
        print(f"{OK} whisperx importable")
        return True
    except ImportError:
        print(f"{WARN} whisperx absent → pip install whisperx")
        print(f"   (fallback whisper standard disponible)")
        return False


def check_torch() -> bool:
    """Vérifie la version de torch et le device."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = "CUDA" if cuda else "CPU"
        print(f"{OK} torch {torch.__version__} — device : {device}")
        return True
    except ImportError:
        print(f"{ERR} torch absent")
        return False


def check_diarization_config() -> bool:
    """Vérifie la configuration de diarisation."""
    if not DIARIZATION.ENABLED:
        print(f"{WARN} DIARIZATION.ENABLED = False")
        return False
    min_s = DIARIZATION.MIN_SPEAKERS
    max_s = DIARIZATION.MAX_SPEAKERS
    print(
        f"{OK} Config diarisation : "
        f"enabled=True, min={min_s}, max={max_s}"
    )
    return True


def test_audio_loading(audio_path: Path) -> bool:
    """Test le chargement audio avec soundfile."""
    if not audio_path.exists():
        print(f"{WARN} Fichier audio absent pour le test : {audio_path}")
        return False
    try:
        import soundfile as sf
        waveform, sr = sf.read(str(audio_path), always_2d=True)
        duration = waveform.shape[0] / sr
        print(
            f"{OK} Audio chargé via soundfile : "
            f"{waveform.shape} | sr={sr} | durée={duration:.1f}s"
        )
        return True
    except Exception as exc:
        print(f"{ERR} Erreur chargement soundfile : {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────

def run_diagnostic(audio_path: Optional[Path] = None):
    """Lance le diagnostic complet."""
    print("=" * 65)
    print("  DIAGNOSTIC DIARISATION PYANNOTE")
    print("=" * 65)

    results = {}

    print("\n🔧 Dépendances Python :")
    results["torch"]     = check_torch()
    results["soundfile"] = check_soundfile()
    results["pyannote"]  = check_pyannote()
    results["whisperx"]  = check_whisperx()

    print("\n🖥️  Outils système :")
    results["ffmpeg"]    = check_ffmpeg()

    print("\n⚙️  Configuration :")
    results["config"]    = check_diarization_config()
    results["token"]     = check_token()

    if results["token"]:
        results["models"] = check_models_access()
    else:
        results["models"] = False

    if audio_path and results["soundfile"]:
        print("\n🎵 Test chargement audio :")
        results["audio"] = test_audio_loading(audio_path)

    # ── Résumé ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    n_ok   = sum(1 for v in results.values() if v)
    n_fail = len(results) - n_ok
    print(f"  Résumé : {n_ok}/{len(results)} OK | {n_fail} problème(s)")
    print("=" * 65)

    if not results.get("models", False):
        print(
            "\n⚡ ACTION REQUISE :\n"
            "   Visitez ces 4 URLs et acceptez les conditions :\n"
        )
        for model_id, url in REQUIRED_MODELS:
            print(f"   {url}")

    if not results.get("ffmpeg", False):
        print(
            "\n⚡ ACTION REQUISE :\n"
            "   Installez FFmpeg : winget install Gyan.FFmpeg\n"
            "   Puis redémarrez votre terminal PowerShell."
        )

    return n_fail == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default=None,
                        help="Chemin optionnel d'un fichier WAV pour tester le chargement")
    args = parser.parse_args()

    audio = Path(args.audio) if args.audio else None
    ok    = run_diagnostic(audio)
    sys.exit(0 if ok else 1)
