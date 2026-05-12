"""Módulo de Dataset customizado para áudio"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple
from .preprocessing import AudioProcessor
from .features import FeatureExtractor


def collate_fn_pad(batch):
    """Collate function com padding automático"""
    features_list = [item[0] for item in batch if item[0] is not None]
    labels_list = [item[1] for item in batch if item[0] is not None]

    if not features_list:
        return None, None

    max_time = max(f.shape[2] for f in features_list)
    padded = []

    for f in features_list:
        if f.shape[2] < max_time:
            pad = torch.nn.functional.pad(f, (0, max_time - f.shape[2]), mode='constant', value=0)
            padded.append(pad.squeeze(0))
        else:
            padded.append(f.squeeze(0))

    features = torch.stack(padded, dim=0)
    features = features.unsqueeze(1)
    labels = torch.LongTensor(labels_list)

    return features, labels


class AudioDataset(Dataset):
    """Dataset customizado para dados de áudio"""

    def __init__(self,
                 audio_files: List[str],
                 labels: List[int],
                 sr: int = 16000,
                 n_mels: int = 128,
                 n_mfcc: int = 40,
                 feature_type: str = 'mel',
                 augmentation: bool = False):
        """
        Inicializa dataset de áudio

        Args:
            audio_files: Lista de caminhos de áudio
            labels: Lista de labels (0=real, 1=fake)
            sr: Sample rate
            n_mels: Número de mel bins
            n_mfcc: Número de coeficientes MFCC
            feature_type: 'mel', 'mfcc', ou 'spec'
            augmentation: Aplicar data augmentation
        """
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.feature_type = feature_type
        self.augmentation = augmentation

        self.processor = AudioProcessor(sr=sr)
        self.extractor = FeatureExtractor(sr=sr, n_mfcc=n_mfcc, n_mels=n_mels)

        assert len(audio_files) == len(labels), "Mismatch between files and labels"

    def __len__(self) -> int:
        """Retorna tamanho do dataset"""
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna um item do dataset"""
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        audio, sr = self.processor.load_audio(audio_path, sr=self.sr)

        if audio is None:
            return None, None

        audio = self.processor.normalize_audio(audio)

        if self.augmentation and np.random.random() > 0.5:
            if np.random.random() > 0.5:
                audio = self.processor.pitch_shift(audio, self.sr, n_steps=np.random.randint(-2, 3))
            else:
                audio = self.processor.time_stretch(audio, rate=np.random.uniform(0.9, 1.1))

        if self.feature_type == 'mel':
            features = self.extractor.extract_mel_spectrogram(audio)
        elif self.feature_type == 'mfcc':
            features = self.extractor.extract_mfcc(audio)
        elif self.feature_type == 'delta_mfcc':
            features = self.extractor.extract_delta_mfcc(audio)
        else:
            features = self.extractor.extract_spectrogram(audio)

        features = self.extractor.normalize_features(features)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        label_tensor = torch.LongTensor([label])[0]

        return features_tensor, label_tensor


class AudioDatasetWithAugmentation(Dataset):
    """Dataset com múltiplos tipos de augmentação"""

    def __init__(self,
                 audio_files: List[str],
                 labels: List[int],
                 sr: int = 16000,
                 n_mels: int = 128,
                 feature_type: str = 'mel',
                 augmentations: List[str] = None):
        """
        Inicializa dataset com augmentação avançada

        Args:
            audio_files: Lista de caminhos
            labels: Lista de labels
            sr: Sample rate
            n_mels: Número de mel bins
            feature_type: Tipo de feature
            augmentations: Lista de técnicas de augmentação
        """
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.feature_type = feature_type

        if augmentations is None:
            augmentations = ['pitch_shift', 'time_stretch', 'gaussian_noise']

        self.augmentations = augmentations
        self.processor = AudioProcessor(sr=sr)
        self.extractor = FeatureExtractor(sr=sr, n_mels=n_mels)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna item com augmentação aleatória"""
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        audio, _ = self.processor.load_audio(audio_path, sr=self.sr)

        if audio is None:
            return None, None

        audio = self.processor.normalize_audio(audio)

        # Aplicar augmentação aleatória
        if np.random.random() > 0.5:
            aug = np.random.choice(self.augmentations)

            if aug == 'pitch_shift':
                audio = self.processor.pitch_shift(audio, self.sr, n_steps=np.random.randint(-2, 3))
            elif aug == 'time_stretch':
                audio = self.processor.time_stretch(audio, rate=np.random.uniform(0.9, 1.1))
            elif aug == 'gaussian_noise':
                audio = self.processor.add_gaussian_noise(audio, snr_db=np.random.uniform(15, 30))

        # Extrair features
        if self.feature_type == 'mel':
            features = self.extractor.extract_mel_spectrogram(audio)
        else:
            features = self.extractor.extract_mfcc(audio)

        features = self.extractor.normalize_features(features)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        return features_tensor, torch.LongTensor([label])[0]


class PaddedAudioDataset(Dataset):
    """Dataset com padding para terem mesmo tamanho"""

    def __init__(self,
                 audio_files: List[str],
                 labels: List[int],
                 sr: int = 16000,
                 n_mels: int = 128,
                 max_frames: int = 300):
        """
        Inicializa dataset com padding

        Args:
            audio_files: Lista de caminhos
            labels: Lista de labels
            sr: Sample rate
            n_mels: Número de mel bins
            max_frames: Máximo de frames (para padding)
        """
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames

        self.processor = AudioProcessor(sr=sr)
        self.extractor = FeatureExtractor(sr=sr, n_mels=n_mels)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna item com padding"""
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        audio, _ = self.processor.load_audio(audio_path, sr=self.sr)

        if audio is None:
            # Retornar tensor com padding
            padded = np.zeros((self.n_mels, self.max_frames))
            return torch.FloatTensor(padded).unsqueeze(0), torch.LongTensor([label])[0]

        audio = self.processor.normalize_audio(audio)

        # Extrair features
        features = self.extractor.extract_mel_spectrogram(audio)

        # Normalizar
        features = self.extractor.normalize_features(features)

        # Fazer padding ou truncar
        if features.shape[1] < self.max_frames:
            # Padding (espelhamento)
            pad_amount = self.max_frames - features.shape[1]
            padded = np.pad(features, ((0, 0), (0, pad_amount)),
                          mode='reflect')
        else:
            # Truncar
            padded = features[:, :self.max_frames]

        features_tensor = torch.FloatTensor(padded).unsqueeze(0)

        return features_tensor, torch.LongTensor([label])[0]


def collate_fn_with_padding(batch):
    """
    Função customizada para colate de batches

    Maneja tensores de tamanhos diferentes
    """
    features_list = []
    labels_list = []

    for features, label in batch:
        if features is not None:
            features_list.append(features)
            labels_list.append(label)

    if len(features_list) == 0:
        return None, None

    # Pad features para mesmo tamanho dentro do batch
    max_frames = max([f.shape[-1] for f in features_list])

    padded_features = []
    for f in features_list:
        if f.shape[-1] < max_frames:
            pad_amount = max_frames - f.shape[-1]
            f_padded = torch.nn.functional.pad(f, (0, pad_amount))
        else:
            f_padded = f

        padded_features.append(f_padded)

    features_batch = torch.cat(padded_features, dim=0)
    labels_batch = torch.stack(labels_list)

    return features_batch, labels_batch


if __name__ == "__main__":
    # Teste do dataset
    from pathlib import Path

    # Criar dados dummy
    sr = 16000
    duration = 2
    t = np.linspace(0, duration, int(sr * duration))

    # Salvar alguns áudios de teste
    test_dir = Path('test_audio')
    test_dir.mkdir(exist_ok=True)

    for i in range(5):
        audio = np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))
        import soundfile as sf
        try:
            sf.write(test_dir / f'real_{i}.wav', audio, sr)
        except:
            # Usar librosa se soundfile não estiver disponível
            import librosa
            librosa.output.write_wav(test_dir / f'real_{i}.wav', audio, sr)

    # Criar dataset
    files = list((test_dir).glob('*.wav'))
    labels = [0] * len(files)

    dataset = AudioDataset(
        audio_files=[str(f) for f in files],
        labels=labels,
        feature_type='mel'
    )

    print(f"[OK] Dataset criado com {len(dataset)} samples")

    # Carregar um sample
    features, label = dataset[0]
    print(f"[OK] Sample shape: {features.shape}, Label: {label}")
