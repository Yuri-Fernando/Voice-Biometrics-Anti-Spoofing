"""Módulo de Pré-processamento de Áudio"""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class AudioProcessor:
    """Processador de áudio com extração de features"""

    def __init__(self, sr: int = 16000):
        """
        Inicializa processador de áudio

        Args:
            sr: Sample rate (frequência de amostragem)
        """
        self.sr = sr

    @staticmethod
    def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Carrega áudio e converte para mono a 16kHz

        Args:
            file_path: Caminho do arquivo de áudio
            sr: Sample rate desejado

        Returns:
            Tupla (audio_array, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=sr, mono=True)
            return audio, sr
        except Exception as e:
            print(f"[ERROR] Erro ao carregar {file_path}: {e}")
            return None, None

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        """
        Normaliza áudio entre -1 e 1

        Args:
            audio: Array de áudio
            target_level: Nível alvo de normalização

        Returns:
            Áudio normalizado
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (target_level / max_val)
        return audio

    @staticmethod
    def apply_preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Aplica filtro de pré-ênfase
        Acentua altas frequências

        Args:
            audio: Array de áudio
            coef: Coeficiente de pré-ênfase

        Returns:
            Áudio com pré-ênfase aplicada
        """
        audio_emphasized = np.append(audio[0], audio[1:] - coef * audio[:-1])
        return audio_emphasized

    @staticmethod
    def frame_audio(audio: np.ndarray, frame_length: int = 2048,
                    hop_length: int = 512) -> np.ndarray:
        """
        Divide áudio em frames (janelas)

        Args:
            audio: Array de áudio
            frame_length: Tamanho do frame
            hop_length: Deslocamento entre frames

        Returns:
            Matrix de frames (n_frames, frame_length)
        """
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        frames = np.zeros((n_frames, frame_length))

        for i in range(n_frames):
            start = i * hop_length
            frames[i, :] = audio[start:start + frame_length]

        return frames

    @staticmethod
    def apply_window(frames: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
        """
        Aplica janela (windowing) aos frames

        Args:
            frames: Matrix de frames
            window_type: Tipo de janela ('hamming', 'hann', 'blackman')

        Returns:
            Frames com janela aplicada
        """
        if window_type == 'hamming':
            window = np.hamming(frames.shape[1])
        elif window_type == 'hann':
            window = np.hanning(frames.shape[1])
        elif window_type == 'blackman':
            window = np.blackman(frames.shape[1])
        else:
            window = np.ones(frames.shape[1])

        return frames * window

    @staticmethod
    def remove_silence(audio: np.ndarray, sr: int,
                      threshold_db: float = -40) -> np.ndarray:
        """
        Remove silêncio do áudio

        Args:
            audio: Array de áudio
            sr: Sample rate
            threshold_db: Limiar de decibéis

        Returns:
            Áudio sem silêncio
        """
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        mask = S_db > threshold_db
        frames_to_keep = np.any(mask, axis=0)

        frame_length = len(audio) // S_db.shape[1]
        samples_to_keep = np.repeat(frames_to_keep, frame_length)

        return audio[:len(samples_to_keep)][samples_to_keep]

    @staticmethod
    def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
        """
        Faz pitch shift no áudio (data augmentation)

        Args:
            audio: Array de áudio
            sr: Sample rate
            n_steps: Número de semitons para deslocar

        Returns:
            Áudio com pitch modificado
        """
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    @staticmethod
    def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Altera a velocidade do áudio (data augmentation)

        Args:
            audio: Array de áudio
            rate: Taxa de alteração (1.0 = original)

        Returns:
            Áudio com velocidade alterada
        """
        return librosa.effects.time_stretch(audio, rate=rate)

    @staticmethod
    def add_gaussian_noise(audio: np.ndarray, snr_db: float = 20) -> np.ndarray:
        """
        Adiciona ruído gaussiano ao áudio (data augmentation)

        Args:
            audio: Array de áudio
            snr_db: Signal-to-Noise Ratio em dB

        Returns:
            Áudio com ruído adicionado
        """
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise

    @staticmethod
    def extract_statistical_features(features: np.ndarray) -> np.ndarray:
        """
        Extrai features estatísticas (média, desvio padrão)

        Args:
            features: Matrix de features (n_features, n_frames)

        Returns:
            Vetor de features estatísticas
        """
        mean = np.mean(features.T, axis=0)
        std = np.std(features.T, axis=0)
        return np.concatenate([mean, std])


def load_audio_files(directory: str, pattern: str = "*.wav") -> list:
    """
    Carrega todos os arquivos de áudio de um diretório

    Args:
        directory: Diretório com arquivos
        pattern: Padrão de arquivo (ex: *.wav)

    Returns:
        Lista de caminhos de arquivo
    """
    path = Path(directory)
    return list(path.glob(pattern))


if __name__ == "__main__":
    # Teste rápido
    processor = AudioProcessor(sr=16000)

    # Criar áudio de teste
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 200 * t)

    # Normalizar
    audio_norm = processor.normalize_audio(audio)
    print(f"[OK] Áudio normalizado: min={audio_norm.min():.3f}, max={audio_norm.max():.3f}")

    # Aplicar pré-ênfase
    audio_emph = processor.apply_preemphasis(audio_norm)
    print(f"[OK] Pré-ênfase aplicada: shape={audio_emph.shape}")

    # Remover silêncio
    audio_no_silence = processor.remove_silence(audio_norm, sr)
    print(f"[OK] Silêncio removido: shape={audio_no_silence.shape}")
