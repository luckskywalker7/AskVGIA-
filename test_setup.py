"""
Test rapide de la configuration
"""

import torch
import whisper
import ollama
from config.settings import (
    UPLOADS_DIR, AUDIO_DIR, CHROMA_DIR,
    WHISPER, OLLAMA
)


def test_setup():
    """Test de la configuration"""

    print("=" * 80)
    print("TEST CONFIGURATION")
    print("=" * 80)

    # 1. Directories
    print("\n1. Directories:")
    print(f"   ✅ UPLOADS_DIR: {UPLOADS_DIR}")
    print(f"   ✅ AUDIO_DIR: {AUDIO_DIR}")
    print(f"   ✅ CHROMA_DIR: {CHROMA_DIR}")

    # 2. PyTorch
    print("\n2. PyTorch:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 3. Whisper
    print("\n3. Whisper:")
    print(f"   Model: {WHISPER.MODEL_SIZE}")
    try:
        model = whisper.load_model(WHISPER.MODEL_SIZE)
        print(f"   ✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # 4. Ollama
    print("\n4. Ollama:")
    try:
        models = ollama.list()
        print(f"   ✅ Ollama connecté")

        # Vérifier les modèles
        model_names = []
        if hasattr(models, 'models'):
            model_names = [m.model if hasattr(m, 'model') else m.name
                           for m in models.models]
        elif isinstance(models, dict) and 'models' in models:
            model_names = [m.get('model', m.get('name', ''))
                           for m in models['models']]

        print(f"   Modèles installés: {len(model_names)}")

        # Vérifier nomic-embed-text
        has_embed = any('nomic-embed-text' in name for name in model_names)
        has_llama = any('llama3.1' in name for name in model_names)

        if has_embed:
            print(f"   ✅ nomic-embed-text trouvé")
        else:
            print(f"   ❌ nomic-embed-text manquant")
            print(f"      Installer avec: ollama pull nomic-embed-text")

        if has_llama:
            print(f"   ✅ llama3.1:8b trouvé")
        else:
            print(f"   ❌ llama3.1:8b manquant")
            print(f"      Installer avec: ollama pull llama3.1:8b")

    except Exception as e:
        print(f"   ❌ Erreur Ollama: {e}")

    print("\n" + "=" * 80)
    print("✅ TEST TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    test_setup()
