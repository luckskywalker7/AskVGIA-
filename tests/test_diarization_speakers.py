# tests/test_diarization_speakers.py
"""
Test diarisation — Pipeline complet avec diagnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage :
    python tests/test_diarization_speakers.py
    python tests/test_diarization_speakers.py --video data/uploads/videoplayback.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_diarization")

from config.settings import DIARIZATION, UPLOADS_DIR
from core.transcription import VideoTranscriber

from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────

def print_separator(char="═", width=70):
    print(char * width)


def print_speaker_stats(result):
    """Affiche les statistiques par speaker."""
    if not result.has_diarization:
        print("\n⚠️  Pas de diarisation — tous les segments = SPEAKER_00")
        return

    print(f"\n🗣️  ANALYSE PAR SPEAKER ({len(result.speakers)} détectés)")
    print_separator("─")

    for spk in result.speakers:
        info = result.diarization_info.get(spk, {})
        n    = info.get("n_segments", 0)
        dur  = info.get("duration_s", 0.0)
        pct  = info.get("percentage", 0.0)
        idx  = result.speakers.index(spk) + 1

        print(f"\n  Speaker {idx} ({spk})")
        print(f"    ⏱️  Durée     : {dur:.1f}s  ({pct:.1f}%)")
        print(f"    💬 Segments  : {n}")
        if dur > 0:
            words = sum(len(s.text.split()) for s in result.segments if s.speaker == spk)
            wpm   = words / (dur / 60)
            print(f"    📝 Mots/min : {wpm:.0f}")


def print_dialogue(result, max_segments: int = 25):
    """Affiche le dialogue formaté par speaker."""
    print(f"\n💬 DIALOGUE (premiers {max_segments} segments)")
    print_separator("─")

    current_spk = None
    for seg in result.segments[:max_segments]:
        if seg.speaker != current_spk:
            current_spk  = seg.speaker
            spk_idx = result.speakers.index(seg.speaker) + 1 if seg.speaker in result.speakers else "?"
            print(f"\n  🗣️  {seg.speaker_label} [{seg.timestamp_start}]")

        preview = seg.text[:90]
        if len(seg.text) > 90:
            preview += "…"
        print(f"     {preview}")


def print_speaker_timeline(result, width: int = 60):
    """
    Affiche une timeline ASCII des speakers.
    Exemple :
      00:00 ████░░░░███░░████  19:00
             S1  S2  S1  S2
    """
    if not result.has_diarization or result.duration <= 0:
        return

    print(f"\n📊 TIMELINE DES SPEAKERS")
    print_separator("─")

    duration = result.duration
    bins     = [[] for _ in range(width)]

    for seg in result.segments:
        s_bin = int(seg.start_time / duration * width)
        e_bin = int(seg.end_time   / duration * width)
        spk_idx = result.speakers.index(seg.speaker) if seg.speaker in result.speakers else 0

        for b in range(min(s_bin, width - 1), min(e_bin + 1, width)):
            bins[b].append(spk_idx)

    symbols = ["█", "▓", "░", "▒", "■", "□"]
    line    = ""
    for b in bins:
        if not b:
            line += " "
        else:
            dominant = max(set(b), key=b.count)
            line    += symbols[dominant % len(symbols)]

    # Labels temps
    ts_start = "00:00"
    ts_end   = f"{int(duration//60):02d}:{int(duration%60):02d}"

    print(f"  {ts_start}  {line}  {ts_end}")

    # Légende
    legend = "  Légende : " + " | ".join(
        f"{symbols[i % len(symbols)]}={result.speakers[i]}"
        for i in range(len(result.speakers))
    )
    print(legend)


def test_full_pipeline(video_path: Path):
    """Test complet du pipeline."""
    print_separator()
    print(f"  TEST DIARISATION — {video_path.name}")
    print_separator()
    print(f"  Token HF   : {DIARIZATION.HF_TOKEN[:12]}…")
    print(f"  Min/Max speakers : {DIARIZATION.MIN_SPEAKERS}/{DIARIZATION.MAX_SPEAKERS}")
    print(f"  Diarisation : {'ON' if DIARIZATION.ENABLED else 'OFF'}")
    print_separator("─")

    # ── Pipeline ──────────────────────────────────────────────────────────
    transcriber = VideoTranscriber(model_size="medium")
    result      = transcriber.process_video(
        video_path=video_path,
        language=  None,
    )

    if result is None:
        print("\n❌ Échec de la transcription")
        return False

    # ── Résultats ─────────────────────────────────────────────────────────
    print_separator()
    print("  RÉSULTATS")
    print_separator()
    print(f"  Langue         : {result.language}")
    print(f"  Durée          : {result.duration:.1f}s  ({result.duration/60:.1f} min)")
    print(f"  Segments       : {result.total_segments}")
    print(f"  Diarisation    : {'✅ OUI' if result.has_diarization else '❌ NON'}")
    print(f"  Speakers       : {len(result.speakers)}  → {result.speakers}")
    print(f"  Temps traitement : {result.processing_time:.1f}s")

    # Stats
    print_speaker_stats(result)

    # Timeline
    print_speaker_timeline(result)

    # Dialogue
    print_dialogue(result)

    print_separator("─")
    if result.has_diarization and len(result.speakers) > 1:
        print(f"\n✅ Diarisation réussie : {len(result.speakers)} speakers identifiés !")
    elif result.has_diarization and len(result.speakers) == 1:
        print(f"\n⚠️  Diarisation OK mais un seul speaker détecté.")
        print(f"   Vérifiez DIARIZATION.MIN_SPEAKERS / MAX_SPEAKERS dans settings.py")
    else:
        print(f"\n❌ Diarisation non disponible.")
        print(f"   Exécutez : python tests/test_diarization_diagnostic.py")

    return result.has_diarization


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Chemin vers la vidéo à tester",
    )
    args = parser.parse_args()

    if args.video:
        video = Path(args.video)
    else:
        # Auto-détection
        videos = list(UPLOADS_DIR.glob("*.mp4")) + list(UPLOADS_DIR.glob("*.avi"))
        if not videos:
            print("❌ Aucune vidéo dans data/uploads/")
            print("   Spécifiez un chemin : --video data/uploads/ma_video.mp4")
            sys.exit(1)
        video = videos[0]
        print(f"📁 Vidéo auto-détectée : {video.name}")

    if not video.exists():
        print(f"❌ Fichier introuvable : {video}")
        sys.exit(1)

    ok = test_full_pipeline(video)
    sys.exit(0 if ok else 1)
