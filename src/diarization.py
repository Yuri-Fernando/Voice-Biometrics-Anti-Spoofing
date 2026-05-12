"""Módulo de Diarização de Locutor"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
from .preprocessing import AudioProcessor
from .features import FeatureExtractor


class SpeakerDiarization:
    """Sistema de diarização (identificar quem falou quando)"""

    def __init__(self, n_speakers: int = 2, sr: int = 16000, n_mels: int = 128):
        """
        Inicializa sistema de diarização

        Args:
            n_speakers: Número esperado de speakers
            sr: Sample rate
            n_mels: Número de mel bins
        """
        self.n_speakers = n_speakers
        self.sr = sr
        self.n_mels = n_mels
        self.processor = AudioProcessor(sr=sr)
        self.extractor = FeatureExtractor(sr=sr, n_mels=n_mels)

    def segment_audio(self, audio: np.ndarray,
                     segment_duration: float = 2.0,
                     overlap: float = 0.5) -> List[Dict]:
        """
        Segmenta áudio em chunks com overlap

        Args:
            audio: Array de áudio
            segment_duration: Duração de cada segmento em segundos
            overlap: Sobreposição (0.0 a 1.0)

        Returns:
            Lista de segmentos com timestamps
        """
        segment_samples = int(segment_duration * self.sr)
        hop_samples = int(segment_samples * (1 - overlap))

        segments = []
        start_idx = 0

        while start_idx + segment_samples <= len(audio):
            segment = audio[start_idx:start_idx + segment_samples]

            start_time = start_idx / self.sr
            end_time = (start_idx + segment_samples) / self.sr

            segments.append({
                'audio': segment,
                'start': start_time,
                'end': end_time,
                'start_sample': start_idx,
                'end_sample': start_idx + segment_samples
            })

            start_idx += hop_samples

        # Último segmento (se houver resto)
        if start_idx < len(audio):
            remaining = audio[start_idx:]
            if len(remaining) > segment_samples * 0.3:  # Mínimo 30%
                segments.append({
                    'audio': remaining,
                    'start': start_idx / self.sr,
                    'end': len(audio) / self.sr,
                    'start_sample': start_idx,
                    'end_sample': len(audio)
                })

        return segments

    def extract_segment_embeddings(self, segments: List[Dict],
                                  model: nn.Module,
                                  device: torch.device) -> np.ndarray:
        """
        Extrai embeddings de cada segmento

        Args:
            segments: Lista de segmentos
            model: Modelo de embedding treinado
            device: Device (cuda/cpu)

        Returns:
            Matrix de embeddings (n_segments, embedding_dim)
        """
        embeddings = []
        model.eval()

        with torch.no_grad():
            for segment in segments:
                audio = segment['audio']

                # Extrair mel-spectrogram
                mel_spec = self.extractor.extract_mel_spectrogram(audio)
                mel_spec = self.extractor.normalize_features(mel_spec)

                # Converter para tensor
                mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

                # Forward pass
                embedding = model(mel_tensor).cpu().numpy()
                embeddings.append(embedding.flatten())

        return np.array(embeddings)

    def cluster_speakers(self, embeddings: np.ndarray,
                        method: str = 'kmeans') -> np.ndarray:
        """
        Agrupa segmentos por locutor usando clustering

        Args:
            embeddings: Matrix de embeddings
            method: 'kmeans' ou 'agglomerative'

        Returns:
            Array de labels (um por segmento)
        """
        if embeddings.shape[0] < self.n_speakers:
            print(f"[WARN]  Menos segmentos ({embeddings.shape[0]}) que speakers ({self.n_speakers})")
            return np.arange(embeddings.shape[0])

        # Normalizar embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=self.n_speakers, random_state=42, n_init=10)
        else:  # agglomerative
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_speakers,
                linkage='ward'
            )

        labels = clusterer.fit_predict(embeddings_scaled)
        return labels

    def smooth_labels(self, labels: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Suaviza mudanças abruptas de speaker (pós-processamento)

        Args:
            labels: Array de labels
            window_size: Tamanho da janela de suavização

        Returns:
            Labels suavizados
        """
        smoothed = labels.copy()

        for i in range(window_size, len(labels) - window_size):
            window = labels[i - window_size:i + window_size + 1]
            # Mode (valor mais frequente)
            unique, counts = np.unique(window, return_counts=True)
            smoothed[i] = unique[np.argmax(counts)]

        return smoothed

    def merge_segments(self, segments: List[Dict],
                      labels: np.ndarray,
                      min_duration: float = 0.5) -> List[Dict]:
        """
        Merge segmentos do mesmo speaker e duração mínima

        Args:
            segments: Lista de segmentos
            labels: Array de labels de speaker
            min_duration: Duração mínima em segundos

        Returns:
            Lista de segmentos merged
        """
        if len(segments) == 0:
            return []

        merged = []
        current_speaker = labels[0]
        current_start = segments[0]['start']
        current_end = segments[0]['end']

        for i in range(1, len(segments)):
            if labels[i] == current_speaker:
                # Continua mesmo speaker
                current_end = segments[i]['end']
            else:
                # Novo speaker - salvar segmento anterior
                duration = current_end - current_start
                if duration >= min_duration:
                    merged.append({
                        'start': current_start,
                        'end': current_end,
                        'speaker': f"Speaker_{current_speaker}",
                        'duration': duration
                    })

                current_speaker = labels[i]
                current_start = segments[i]['start']
                current_end = segments[i]['end']

        # Último segmento
        duration = current_end - current_start
        if duration >= min_duration:
            merged.append({
                'start': current_start,
                'end': current_end,
                'speaker': f"Speaker_{current_speaker}",
                'duration': duration
            })

        return merged

    def diarize(self, audio_path: str, model: nn.Module,
               segment_duration: float = 2.0,
               device: torch.device = None) -> pd.DataFrame:
        """
        Pipeline completo de diarização

        Args:
            audio_path: Caminho do áudio
            model: Modelo de embedding treinado
            segment_duration: Duração dos segmentos
            device: Device (cuda/cpu)

        Returns:
            DataFrame com [start_time, end_time, speaker_id, duration]
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Carregar áudio
        audio, sr = self.processor.load_audio(audio_path, sr=self.sr)

        if audio is None:
            print(f"[ERROR] Erro ao carregar áudio")
            return pd.DataFrame()

        # 2. Segmentar
        segments = self.segment_audio(audio, segment_duration=segment_duration)

        if len(segments) == 0:
            print(f"[ERROR] Nenhum segmento extraído")
            return pd.DataFrame()

        # 3. Extrair embeddings
        embeddings = self.extract_segment_embeddings(segments, model, device)

        # 4. Clustering
        speaker_labels = self.cluster_speakers(embeddings)

        # 5. Suavização
        speaker_labels = self.smooth_labels(speaker_labels)

        # 6. Merge
        diarization_result = self.merge_segments(segments, speaker_labels)

        return pd.DataFrame(diarization_result)

    def compute_speaker_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estatísticas por speaker

        Args:
            df: DataFrame de diarização

        Returns:
            DataFrame com estatísticas
        """
        stats = df.groupby('speaker').agg({
            'duration': ['sum', 'mean', 'min', 'max', 'count'],
            'start': 'min',
            'end': 'max'
        }).round(2)

        return stats


class SpeakerVerification:
    """Sistema de verificação de speaker (1:1)"""

    def __init__(self, sr: int = 16000, n_mels: int = 128):
        """
        Inicializa verificador de speaker

        Args:
            sr: Sample rate
            n_mels: Número de mel bins
        """
        self.sr = sr
        self.n_mels = n_mels
        self.processor = AudioProcessor(sr=sr)
        self.extractor = FeatureExtractor(sr=sr, n_mels=n_mels)
        self.reference_embeddings = {}

    def register_speaker(self, speaker_id: str, audio_paths: List[str],
                        model: nn.Module, device: torch.device) -> bool:
        """
        Registra um speaker com múltiplos áudios

        Args:
            speaker_id: ID do speaker
            audio_paths: Lista de caminhos de áudio
            model: Modelo de embedding
            device: Device

        Returns:
            True se registrado com sucesso
        """
        embeddings = []

        for audio_path in audio_paths:
            audio, _ = self.processor.load_audio(audio_path, sr=self.sr)

            if audio is None:
                print(f"[WARN]  Erro ao carregar {audio_path}")
                continue

            audio = self.processor.normalize_audio(audio)
            mel_spec = self.extractor.extract_mel_spectrogram(audio)
            mel_spec = self.extractor.normalize_features(mel_spec)

            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(mel_tensor).cpu().numpy().flatten()

            embeddings.append(embedding)

        if len(embeddings) > 0:
            # Usar média dos embeddings como referência
            mean_embedding = np.mean(embeddings, axis=0)
            self.reference_embeddings[speaker_id] = mean_embedding
            print(f"[OK] Speaker '{speaker_id}' registrado com {len(embeddings)} amostras")
            return True

        return False

    def verify_speaker(self, audio_path: str, speaker_id: str,
                      model: nn.Module, device: torch.device,
                      threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Verifica se áudio pertence ao speaker

        Args:
            audio_path: Caminho do áudio
            speaker_id: ID do speaker registrado
            model: Modelo de embedding
            device: Device
            threshold: Limiar de similaridade

        Returns:
            Tupla (é_mesmo_speaker, score_similaridade)
        """
        if speaker_id not in self.reference_embeddings:
            print(f"[ERROR] Speaker '{speaker_id}' não registrado")
            return False, 0.0

        # Extrair embedding do novo áudio
        audio, _ = self.processor.load_audio(audio_path, sr=self.sr)

        if audio is None:
            return False, 0.0

        audio = self.processor.normalize_audio(audio)
        mel_spec = self.extractor.extract_mel_spectrogram(audio)
        mel_spec = self.extractor.normalize_features(mel_spec)

        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(mel_tensor).cpu().numpy().flatten()

        # Calcular similaridade coseno
        similarity = np.dot(embedding, self.reference_embeddings[speaker_id]) / (
            np.linalg.norm(embedding) * np.linalg.norm(self.reference_embeddings[speaker_id])
        )

        is_match = similarity >= threshold

        return is_match, similarity


if __name__ == "__main__":
    print("[OK] Módulo diarization.py importado com sucesso!")
