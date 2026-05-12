"""Speech AI System - Source Package"""

# Core modules
from .preprocessing import AudioProcessor
from .features import FeatureExtractor
from .models import CNNAntiSpoofing, SpeakerEmbedding, SpeechAISystem
from .dataset import AudioDataset
from .training import ModelTrainer
from .evaluation import ModelEvaluator
from .diarization import SpeakerDiarization, SpeakerVerification

# Scripts
from .train import train_model
from .predict import AudioInference
from .main import SpeechAIProject

__version__ = "1.0.0"
__all__ = [
    "AudioProcessor",
    "FeatureExtractor",
    "CNNAntiSpoofing",
    "SpeakerEmbedding",
    "SpeechAISystem",
    "AudioDataset",
    "ModelTrainer",
    "ModelEvaluator",
    "SpeakerDiarization",
    "SpeakerVerification",
    "train_model",
    "AudioInference",
    "SpeechAIProject",
]
