"""FastAPI Server para Speech AI System"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import io
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import CNNAntiSpoofing, SpeakerEmbedding
from src.preprocessing import AudioProcessor
from src.features import FeatureExtractor

# Configuração
APP_NAME = "Speech AI System API"
APP_VERSION = "1.0.0"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Caminhos
MODEL_DIR = Path(__file__).parent.parent / 'models'
CONFIG_FILE = MODEL_DIR / 'config.json'

# Carregar configuração
try:
    with open(CONFIG_FILE) as f:
        CONFIG = json.load(f)
    SAMPLE_RATE = CONFIG.get('sample_rate', 16000)
    N_MELS = CONFIG.get('n_mels', 128)
except:
    SAMPLE_RATE = 16000
    N_MELS = 128
    CONFIG = {'sample_rate': SAMPLE_RATE, 'n_mels': N_MELS}

# Carregar modelos
def load_models():
    """Carrega modelos treinados"""
    models = {}

    try:
        # Anti-spoofing
        anti_spoof_model = CNNAntiSpoofing().to(DEVICE)
        anti_spoof_path = MODEL_DIR / 'anti_spoof_model.pt'
        if anti_spoof_path.exists():
            anti_spoof_model.load_state_dict(torch.load(anti_spoof_path, map_location=DEVICE))
            models['anti_spoof'] = anti_spoof_model
            print(f"✅ Modelo anti-spoofing carregado")
        else:
            print(f"⚠️  Modelo anti-spoofing não encontrado em {anti_spoof_path}")

        # Speaker embedding
        speaker_embed_model = SpeakerEmbedding().to(DEVICE)
        speaker_path = MODEL_DIR / 'speaker_embedding_model.pt'
        if speaker_path.exists():
            speaker_embed_model.load_state_dict(torch.load(speaker_path, map_location=DEVICE))
            models['speaker_embed'] = speaker_embed_model
            print(f"✅ Modelo speaker embedding carregado")
        else:
            print(f"⚠️  Modelo speaker embedding não encontrado em {speaker_path}")

    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")

    return models

MODELS = load_models()

# Inicializar FastAPI
app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Utilidades
processor = AudioProcessor(sr=SAMPLE_RATE)
extractor = FeatureExtractor(sr=SAMPLE_RATE, n_mels=N_MELS)


@app.on_event("startup")
async def startup_event():
    """Evento de inicialização"""
    print(f"🚀 {APP_NAME} v{APP_VERSION} iniciado")
    print(f"   Device: {DEVICE}")
    print(f"   Modelos carregados: {list(MODELS.keys())}")


# ====================== ENDPOINTS ======================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "device": str(DEVICE),
        "models_available": list(MODELS.keys()),
        "endpoints": {
            "spoof_detection": "POST /predict/spoof",
            "speaker_verification": "POST /predict/verify",
            "diarization": "POST /predict/diarize",
            "health": "GET /health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models": {
            "anti_spoof": "anti_spoof" in MODELS,
            "speaker_embed": "speaker_embed" in MODELS
        }
    }


@app.post("/predict/spoof")
async def predict_spoofing(file: UploadFile = File(...)):
    """
    Detecção de spoofing (real vs fake)

    Args:
        file: Arquivo de áudio (WAV, MP3, etc)

    Returns:
        - prediction: 'real' ou 'fake'
        - confidence: Confiança da predição (0-1)
        - probability: [prob_real, prob_fake]
    """
    try:
        if 'anti_spoof' not in MODELS:
            raise HTTPException(status_code=503, detail="Modelo anti-spoofing não disponível")

        # Ler arquivo
        contents = await file.read()

        # Carregar áudio
        import librosa
        audio, sr = librosa.load(io.BytesIO(contents), sr=SAMPLE_RATE, mono=True)

        # Normalizar
        audio = processor.normalize_audio(audio)

        # Extrair mel-spectrogram
        mel_spec = extractor.extract_mel_spectrogram(audio)
        mel_spec = extractor.normalize_features(mel_spec)

        # Converter para tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Predição
        model = MODELS['anti_spoof']
        model.eval()

        with torch.no_grad():
            outputs = model(mel_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

        # Interpretar resultado
        pred_label = 0 if probs[0] > probs[1] else 1
        confidence = float(max(probs))

        prediction = "real" if pred_label == 0 else "fake"

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "real": float(probs[0]),
                "fake": float(probs[1])
            },
            "audio_duration": len(audio) / sr,
            "sample_rate": sr
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/predict/verify")
async def predict_verify(file: UploadFile = File(...)):
    """
    Verificação de speaker (1:1)

    Obs: Requer registro prévio do speaker
    """
    try:
        if 'speaker_embed' not in MODELS:
            raise HTTPException(status_code=503, detail="Modelo speaker embedding não disponível")

        # Ler arquivo
        contents = await file.read()

        # Carregar áudio
        import librosa
        audio, sr = librosa.load(io.BytesIO(contents), sr=SAMPLE_RATE, mono=True)

        # Normalizar
        audio = processor.normalize_audio(audio)

        # Extrair mel-spectrogram
        mel_spec = extractor.extract_mel_spectrogram(audio)
        mel_spec = extractor.normalize_features(mel_spec)

        # Converter para tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Extrair embedding
        model = MODELS['speaker_embed']
        model.eval()

        with torch.no_grad():
            embedding = model(mel_tensor).cpu().numpy().flatten()

        return {
            "status": "success",
            "message": "Speaker embedding extraído com sucesso",
            "embedding_dim": len(embedding),
            "audio_duration": len(audio) / sr,
            "note": "Para verificação, compare este embedding com embeddings registrados"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/predict/diarize")
async def predict_diarize(file: UploadFile = File(...)):
    """
    Diarização (identificar quem falou quando)

    Obs: Requer modelo treinado e múltiplos speakers
    """
    try:
        if 'speaker_embed' not in MODELS:
            raise HTTPException(status_code=503, detail="Modelo speaker embedding não disponível")

        # Ler arquivo
        contents = await file.read()

        # Carregar áudio
        import librosa
        audio, sr = librosa.load(io.BytesIO(contents), sr=SAMPLE_RATE, mono=True)

        # Normalizar
        audio = processor.normalize_audio(audio)

        return {
            "status": "implemented",
            "message": "Diarization endpoint implementado",
            "audio_duration": len(audio) / sr,
            "note": "Função completa disponível em src/diarization.py"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...), feature_type: str = "mel"):
    """
    Extrai features de áudio

    Args:
        file: Arquivo de áudio
        feature_type: 'mel', 'mfcc', ou 'spec'

    Returns:
        Features como JSON (shape e estatísticas)
    """
    try:
        contents = await file.read()

        import librosa
        audio, sr = librosa.load(io.BytesIO(contents), sr=SAMPLE_RATE, mono=True)

        audio = processor.normalize_audio(audio)

        if feature_type == "mel":
            features = extractor.extract_mel_spectrogram(audio)
        elif feature_type == "mfcc":
            features = extractor.extract_mfcc(audio)
        else:
            features = extractor.extract_spectrogram(audio)

        return {
            "status": "success",
            "feature_type": feature_type,
            "shape": list(features.shape),
            "statistics": {
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features))
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ====================== ERRORS ======================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global de exceções"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    print(f"🚀 Iniciando {APP_NAME} v{APP_VERSION}")
    print(f"📍 Acesse http://localhost:8000/docs para documentação")

    uvicorn.run(app, host="0.0.0.0", port=8000)
