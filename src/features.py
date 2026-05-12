"""Módulo de Extração de Features de Áudio"""

import librosa
import numpy as np
from typing import Tuple


class FeatureExtractor:
    """Classe para extração de features de áudio"""

    def __init__(self, sr: int = 16000, n_mfcc: int = 40, n_mels: int = 128):
        """
        Inicializa extrator de features

        Args:
            sr: Sample rate
            n_mfcc: Número de coeficientes MFCC
            n_mels: Número de mel bins
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai MFCC (Mel Frequency Cepstral Coefficients)

        MFCC é a feature mais usada em processamento de voz.
        Simula como o ouvido humano processa som.

        Args:
            audio: Array de áudio

        Returns:
            Matrix MFCC (n_mfcc, n_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        return mfcc

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai Log-Mel Spectrogram (RECOMENDADO para deep learning)

        Mel-spectrogram é moderno e funciona melhor com redes neurais.
        128 mel-bins dão boa representação densa de frequência.

        Args:
            audio: Array de áudio

        Returns:
            Matrix Log-Mel Spectrogram (n_mels, n_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel

    def extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai Spectrogram simples

        Spectrogram é a base para todas as outras features.
        Mostra como frequências mudam ao longo do tempo.

        Args:
            audio: Array de áudio

        Returns:
            Matrix Spectrogram (n_fft/2 + 1, n_frames)
        """
        spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        log_spec = librosa.power_to_db(spec, ref=np.max)
        return log_spec

    def extract_delta_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai MFCC + variação (delta)

        Delta captura como as features mudam ao longo do tempo.
        Melhora discriminabilidade.

        Args:
            audio: Array de áudio

        Returns:
            Matrix MFCC + Delta (2*n_mfcc, n_frames)
        """
        mfcc = self.extract_mfcc(audio)
        mfcc_delta = librosa.feature.delta(mfcc)
        return np.vstack([mfcc, mfcc_delta])

    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai Chroma features (info de pitch)

        Útil para reconhecimento de padrões musicais.
        12 bins para as 12 notas da escala cromática.

        Args:
            audio: Array de áudio

        Returns:
            Matrix Chroma (12, n_frames)
        """
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
        return chroma

    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai Zero Crossing Rate (ZCR)

        ZCR mede quantas vezes o sinal cruza zero.
        Alto para voz surdas (fricativas), baixo para vozeadas.

        Args:
            audio: Array de áudio

        Returns:
            Vetor ZCR (n_frames,)
        """
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=512)
        return zcr[0]

    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai Spectral Centroid

        Indica a "cor" do som.
        Sons claros têm centroide alto, sons escuros têm centroide baixo.

        Args:
            audio: Array de áudio

        Returns:
            Vetor Spectral Centroid (n_frames,)
        """
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr, hop_length=512)
        return centroid[0]

    def extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai Spectral Rolloff

        Frequência onde 85% da energia está abaixo.
        Mede a largura do espectro.

        Args:
            audio: Array de áudio

        Returns:
            Vetor Spectral Rolloff (n_frames,)
        """
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr, hop_length=512)
        return rolloff[0]

    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai RMS Energy

        Energia do sinal em cada frame.
        Útil para detecção de atividade de voz (VAD).

        Args:
            audio: Array de áudio

        Returns:
            Vetor RMS Energy (n_frames,)
        """
        rms = librosa.feature.rms(y=audio, hop_length=512)[0]
        return rms

    def extract_tempogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai Tempogram (periodicidade temporal)

        Captura padrões rítmicos.

        Args:
            audio: Array de áudio

        Returns:
            Matrix Tempogram
        """
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sr)
        return tempogram

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normaliza features (média 0, desvio padrão 1)

        Args:
            features: Matrix de features

        Returns:
            Features normalizadas
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-8
        return (features - mean) / std

    def extract_combined_features(self, audio: np.ndarray,
                                 features: list = None) -> dict:
        """
        Extrai múltiplas features e retorna dict

        Args:
            audio: Array de áudio
            features: Lista de features desejadas
                     ['mfcc', 'mel', 'spectral_centroid', 'zero_crossing_rate']

        Returns:
            Dicionário com todas as features
        """
        if features is None:
            features = ['mfcc', 'mel']

        result = {}

        if 'mfcc' in features:
            result['mfcc'] = self.extract_mfcc(audio)

        if 'mel' in features:
            result['mel'] = self.extract_mel_spectrogram(audio)

        if 'spectrogram' in features:
            result['spectrogram'] = self.extract_spectrogram(audio)

        if 'delta_mfcc' in features:
            result['delta_mfcc'] = self.extract_delta_mfcc(audio)

        if 'chroma' in features:
            result['chroma'] = self.extract_chroma(audio)

        if 'zero_crossing_rate' in features:
            result['zero_crossing_rate'] = self.extract_zero_crossing_rate(audio)

        if 'spectral_centroid' in features:
            result['spectral_centroid'] = self.extract_spectral_centroid(audio)

        if 'spectral_rolloff' in features:
            result['spectral_rolloff'] = self.extract_spectral_rolloff(audio)

        if 'rms_energy' in features:
            result['rms_energy'] = self.extract_rms_energy(audio)

        return result

    @staticmethod
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula estatísticas (média, std) das features

        Args:
            features: Matrix de features

        Returns:
            Tupla (mean, std)
        """
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        return mean, std


if __name__ == "__main__":
    # Teste rápido
    extractor = FeatureExtractor(sr=16000, n_mfcc=40, n_mels=128)

    # Criar áudio de teste
    sr = 16000
    duration = 2
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)

    # Extrair features
    mfcc = extractor.extract_mfcc(audio)
    mel = extractor.extract_mel_spectrogram(audio)
    zcr = extractor.extract_zero_crossing_rate(audio)

    print(f"[OK] MFCC shape: {mfcc.shape}")
    print(f"[OK] Mel-spectrogram shape: {mel.shape}")
    print(f"[OK] Zero Crossing Rate shape: {zcr.shape}")
    print(f"[OK] ZCR mean: {zcr.mean():.4f}")
