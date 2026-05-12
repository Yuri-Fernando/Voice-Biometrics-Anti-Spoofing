"""Script de inferência para modelos treinados"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np

from src import (
    AudioProcessor, FeatureExtractor,
    CNNAntiSpoofing, SpeakerEmbedding
)


class AudioInference:
    """Classe para fazer inferências com modelos"""

    def __init__(self, model_dir: str = 'models', device: str = None):
        """
        Inicializa para inferência

        Args:
            model_dir: Diretório com modelos
            device: 'cuda' ou 'cpu'
        """
        self.model_dir = Path(model_dir)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)

        # Carregar config com valores padrão
        config_path = self.model_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                loaded_config = json.load(f)
                self.config = loaded_config
        else:
            self.config = {}

        self.config.setdefault('sample_rate', 16000)
        self.config.setdefault('n_mels', 128)
        self.config.setdefault('n_mfcc', 40)
        self.config.setdefault('device', 'cpu')

        # Inicializar processador
        self.processor = AudioProcessor(sr=self.config['sample_rate'])
        self.extractor = FeatureExtractor(
            sr=self.config['sample_rate'],
            n_mfcc=self.config['n_mfcc'],
            n_mels=self.config['n_mels']
        )

        # Carregar modelos
        self.anti_spoof_model = None
        self.speaker_embed_model = None
        self._load_models()

    def _load_models(self):
        """Carrega modelos disponíveis"""
        # Anti-spoofing
        spoof_path = self.model_dir / 'anti_spoof_model.pt'
        if spoof_path.exists():
            self.anti_spoof_model = CNNAntiSpoofing().to(self.device)
            self.anti_spoof_model.load_state_dict(
                torch.load(spoof_path, map_location=self.device)
            )
            self.anti_spoof_model.eval()
            print(f"[OK] Modelo anti-spoofing carregado")

        # Speaker embedding
        embed_path = self.model_dir / 'speaker_embedding_model.pt'
        if embed_path.exists():
            self.speaker_embed_model = SpeakerEmbedding().to(self.device)
            self.speaker_embed_model.load_state_dict(
                torch.load(embed_path, map_location=self.device)
            )
            self.speaker_embed_model.eval()
            print(f"[OK] Modelo speaker embedding carregado")

    def predict_spoofing(self, audio_path: str) -> dict:
        """
        Prediz se áudio é real ou fake

        Args:
            audio_path: Caminho do áudio

        Returns:
            Dicionário com resultado
        """
        if self.anti_spoof_model is None:
            return {'error': 'Modelo anti-spoofing não carregado'}

        # Carregar áudio
        audio, sr = self.processor.load_audio(audio_path, sr=self.config['sample_rate'])

        if audio is None:
            return {'error': f'Não conseguiu carregar {audio_path}'}

        # Processar
        audio = self.processor.normalize_audio(audio)
        mel_spec = self.extractor.extract_mel_spectrogram(audio)
        mel_spec = self.extractor.normalize_features(mel_spec)

        # Tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)

        # Predição
        with torch.no_grad():
            outputs = self.anti_spoof_model(mel_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

        prediction = 'real' if probs[0] > probs[1] else 'fake'
        confidence = float(max(probs))

        return {
            'prediction': prediction,
            'confidence': confidence,
            'prob_real': float(probs[0]),
            'prob_fake': float(probs[1]),
            'audio_duration': len(audio) / sr,
            'sample_rate': sr
        }

    def extract_embedding(self, audio_path: str) -> dict:
        """
        Extrai embedding de voz

        Args:
            audio_path: Caminho do áudio

        Returns:
            Dicionário com embedding
        """
        if self.speaker_embed_model is None:
            return {'error': 'Modelo speaker embedding não carregado'}

        # Carregar áudio
        audio, sr = self.processor.load_audio(audio_path, sr=self.config['sample_rate'])

        if audio is None:
            return {'error': f'Não conseguiu carregar {audio_path}'}

        # Processar
        audio = self.processor.normalize_audio(audio)
        mel_spec = self.extractor.extract_mel_spectrogram(audio)
        mel_spec = self.extractor.normalize_features(mel_spec)

        # Tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)

        # Embedding
        with torch.no_grad():
            embedding = self.speaker_embed_model(mel_tensor)[0].cpu().numpy()

        return {
            'embedding': embedding.tolist(),
            'embedding_dim': len(embedding),
            'audio_duration': len(audio) / sr,
            'norm': float(np.linalg.norm(embedding))
        }

    def compare_embeddings(self, embedding1: np.ndarray,
                          embedding2: np.ndarray) -> dict:
        """
        Compara dois embeddings

        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding

        Returns:
            Dicionário com similaridade
        """
        # Normalizar
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        # Similaridade coseno
        cosine_sim = np.dot(e1, e2)

        # Distância euclidiana
        euclidean_dist = np.linalg.norm(e1 - e2)

        return {
            'cosine_similarity': float(cosine_sim),
            'euclidean_distance': float(euclidean_dist),
            'match_threshold_0.8': cosine_sim >= 0.8,
            'match_threshold_0.9': cosine_sim >= 0.9,
        }


def main():
    """Script principal"""
    parser = argparse.ArgumentParser(description='Fazer predições com modelos treinados')
    parser.add_argument('--audio', type=str, required=True,
                       help='Caminho do arquivo de áudio')
    parser.add_argument('--task', type=str, default='spoof',
                       choices=['spoof', 'embedding', 'compare'],
                       help='Tarefa a executar')
    parser.add_argument('--audio2', type=str,
                       help='Segundo áudio (para comparação)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Diretório com modelos')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Inicializar
    inference = AudioInference(model_dir=args.model_dir, device=args.device)

    print(f"\n{'='*70}")
    print(f"🎵 AUDIO INFERENCE")
    print(f"{'='*70}\n")

    # Executar tarefa
    if args.task == 'spoof':
        print(f"[SEARCH] Detectando spoofing: {args.audio}")
        result = inference.predict_spoofing(args.audio)

        if 'error' in result:
            print(f"[ERROR] {result['error']}")
        else:
            print(f"\n[OK] Resultado:")
            print(f"   Predição: {result['prediction'].upper()}")
            print(f"   Confiança: {result['confidence']*100:.1f}%")
            print(f"   Prob Real: {result['prob_real']:.4f}")
            print(f"   Prob Fake: {result['prob_fake']:.4f}")

    elif args.task == 'embedding':
        print(f"🗣️  Extraindo embedding: {args.audio}")
        result = inference.extract_embedding(args.audio)

        if 'error' in result:
            print(f"[ERROR] {result['error']}")
        else:
            print(f"\n[OK] Embedding extraído:")
            print(f"   Dimensão: {result['embedding_dim']}")
            print(f"   Norma L2: {result['norm']:.4f}")
            print(f"   Duração: {result['audio_duration']:.2f}s")

    elif args.task == 'compare':
        if not args.audio2:
            print("[ERROR] Especifique --audio2 para comparação")
            return

        print(f"🔄 Comparando embeddings:")
        print(f"   Audio 1: {args.audio}")
        print(f"   Audio 2: {args.audio2}\n")

        result1 = inference.extract_embedding(args.audio)
        result2 = inference.extract_embedding(args.audio2)

        if 'error' in result1 or 'error' in result2:
            print("[ERROR] Erro ao extrair embeddings")
            return

        embedding1 = np.array(result1['embedding'])
        embedding2 = np.array(result2['embedding'])

        comparison = inference.compare_embeddings(embedding1, embedding2)

        print(f"[OK] Resultado:")
        print(f"   Similaridade Coseno: {comparison['cosine_similarity']:.4f}")
        print(f"   Distância Euclidiana: {comparison['euclidean_distance']:.4f}")
        print(f"   Match (threshold 0.8): {comparison['match_threshold_0.8']}")
        print(f"   Match (threshold 0.9): {comparison['match_threshold_0.9']}")

    print()


if __name__ == "__main__":
    main()
