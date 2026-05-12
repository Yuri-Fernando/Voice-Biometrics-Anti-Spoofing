"""Speech AI System - Script Principal
Sistema completo de processamento de voz com anti-spoofing, speaker verification e diarização
"""

import sys
import os
from pathlib import Path
import json
import argparse
import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src import (
    AudioProcessor, FeatureExtractor, AudioDataset,
    CNNAntiSpoofing, SpeakerEmbedding, SpeechAISystem,
    ModelTrainer, ModelEvaluator, SpeakerDiarization
)


class SpeechAIProject:
    """Classe principal do projeto"""

    def __init__(self, config_path: str = None):
        """
        Inicializa projeto

        Args:
            config_path: Caminho do arquivo de configuração (JSON)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()

        self._create_directories()

        self.processor = AudioProcessor(sr=self.config['sample_rate'])
        self.extractor = FeatureExtractor(
            sr=self.config['sample_rate'],
            n_mfcc=self.config['n_mfcc'],
            n_mels=self.config['n_mels']
        )

        print(f"Projeto inicializado")
        print(f"Device: {self.device}")
        print(f"Sample Rate: {self.config['sample_rate']} Hz")

    def _default_config(self) -> dict:
        """Retorna configuração padrão"""
        return {
            'sample_rate': 16000,
            'n_mfcc': 40,
            'n_mels': 128,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 10,
            'feature_type': 'mel',
            'test_split': 0.2,
            'val_split': 0.1,
        }

    def _create_directories(self):
        """Cria estrutura de diretórios"""
        dirs = [
            'data/raw',
            'data/processed',
            'models',
            'results',
            'logs'
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def generate_dummy_data(self, n_real: int = 20, n_fake: int = 20,
                           duration: float = 2.0):
        """
        Gera dados de áudio dummy para teste

        Args:
            n_real: Número de áudios reais
            n_fake: Número de áudios fake
            duration: Duração de cada áudio em segundos
        """
        print(f"\nGerando dados dummy...")

        sr = self.config['sample_rate']
        t = np.linspace(0, duration, int(sr * duration))

        for i in range(n_real):
            audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
            audio += 0.1 * np.random.randn(len(audio))
            audio = self.processor.normalize_audio(audio)
            audio = (audio * 32767).astype(np.int16)

            filepath = f'data/raw/real_{i}.wav'
            wavfile.write(filepath, sr, audio)

        print(f"{n_real} áudios reais salvos")

        for i in range(n_fake):
            audio = np.sin(2 * np.pi * 250 * t) + 0.3 * np.sin(2 * np.pi * 350 * t)
            audio = self.processor.normalize_audio(audio)
            audio = (audio * 32767).astype(np.int16)

            filepath = f'data/raw/fake_{i}.wav'
            wavfile.write(filepath, sr, audio)

        print(f"{n_fake} áudios fake salvos")

    def prepare_dataset(self, data_dir: str = 'data/raw') -> tuple:
        """
        Prepara dataset para treinamento

        Args:
            data_dir: Diretório com áudios

        Returns:
            Tupla (train_loader, val_loader)
        """
        print(f"\nPreparando dataset...")

        audio_files = []
        labels = []

        for audio_file in Path(data_dir).glob('*.wav'):
            if 'real' in audio_file.name:
                labels.append(0)
            elif 'fake' in audio_file.name:
                labels.append(1)
            else:
                continue

            audio_files.append(str(audio_file))

        print(f"{len(audio_files)} arquivos carregados")
        print(f"  Reais: {sum(1 for l in labels if l == 0)}")
        print(f"  Fakes: {sum(1 for l in labels if l == 1)}")

        dataset = AudioDataset(
            audio_files=audio_files,
            labels=labels,
            sr=self.config['sample_rate'],
            n_mels=self.config['n_mels'],
            feature_type=self.config['feature_type'],
            augmentation=True
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']
        )

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        return train_loader, val_loader

    def train_anti_spoofing(self, train_loader, val_loader):
        """
        Treina modelo anti-spoofing

        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação

        Returns:
            Modelo treinado
        """
        print(f"\n{'='*60}")
        print(f"TREINANDO MODELO ANTI-SPOOFING")
        print(f"{'='*60}")

        model = CNNAntiSpoofing().to(self.device)
        trainer = ModelTrainer(model, device=self.device, lr=self.config['learning_rate'])

        history = trainer.train(
            train_loader,
            val_loader,
            epochs=self.config['epochs'],
            save_dir='models'
        )

        trainer.save_model('models/anti_spoof_model.pt')

        return model, history

    def train_speaker_embedding(self, train_loader, val_loader):
        """
        Treina modelo de speaker embedding

        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação

        Returns:
            Modelo treinado
        """
        print(f"\n{'='*60}")
        print(f"TREINANDO MODELO SPEAKER EMBEDDING")
        print(f"{'='*60}")

        model = SpeakerEmbedding(embedding_dim=256).to(self.device)
        trainer = ModelTrainer(model, device=self.device, lr=self.config['learning_rate'])

        history = trainer.train(
            train_loader,
            val_loader,
            epochs=self.config['epochs'],
            save_dir='models'
        )

        trainer.save_model('models/speaker_embedding_model.pt')

        return model, history

    def evaluate_model(self, model, val_loader):
        """
        Avalia modelo

        Args:
            model: Modelo a avaliar
            val_loader: DataLoader de validação

        Returns:
            Dicionário de métricas
        """
        print(f"\n{'='*60}")
        print(f"AVALIANDO MODELO")
        print(f"{'='*60}")

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        trainer = ModelTrainer(model, device=self.device)
        val_loss, val_acc, val_preds, val_labels = trainer.evaluate(val_loader, criterion)

        all_scores = []
        with torch.no_grad():
            for features, labels in val_loader:
                if features is None:
                    continue
                features = features.to(self.device)
                outputs = model(features)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_scores.append(probs.cpu().numpy())

        all_scores = np.vstack(all_scores)

        evaluator = ModelEvaluator()
        metrics = evaluator.comprehensive_evaluation(
            np.array(val_labels),
            np.array(val_preds),
            all_scores
        )

        evaluator.print_metrics(metrics)

        return metrics

    def visualize_features(self, data_dir: str = 'data/raw'):
        """
        Visualiza features de áudios reais vs fake

        Args:
            data_dir: Diretório com áudios
        """
        print(f"\nVisualizando features...")

        real_files = list(Path(data_dir).glob('real_*.wav'))
        fake_files = list(Path(data_dir).glob('fake_*.wav'))

        if len(real_files) == 0 or len(fake_files) == 0:
            print("Arquivos não encontrados")
            return

        audio_real, _ = self.processor.load_audio(str(real_files[0]), sr=self.config['sample_rate'])
        audio_fake, _ = self.processor.load_audio(str(fake_files[0]), sr=self.config['sample_rate'])

        mel_real = self.extractor.extract_mel_spectrogram(audio_real)
        mel_fake = self.extractor.extract_mel_spectrogram(audio_fake)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        img1 = axes[0].imshow(mel_real, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Mel-Spectrogram (Real)')
        axes[0].set_xlabel('Tempo')
        axes[0].set_ylabel('Mel Frequency')
        plt.colorbar(img1, ax=axes[0])

        img2 = axes[1].imshow(mel_fake, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Mel-Spectrogram (Fake)')
        axes[1].set_xlabel('Tempo')
        axes[1].set_ylabel('Mel Frequency')
        plt.colorbar(img2, ax=axes[1])

        plt.tight_layout()
        plt.savefig('results/features_comparison.png', dpi=300)
        print(f"Salvo em results/features_comparison.png")

    def save_config(self, path: str = 'models/config.json'):
        """Salva configuração"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuração salva em {path}")


def main():
    """Script principal"""
    parser = argparse.ArgumentParser(description='Speech AI System')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'demo'],
                       help='Modo de execução')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Diretório com dados')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamanho do batch')

    args = parser.parse_args()

    project = SpeechAIProject()

    project.config['epochs'] = args.epochs
    project.config['batch_size'] = args.batch_size

    if args.mode == 'train':
        project.generate_dummy_data(n_real=15, n_fake=15)
        train_loader, val_loader = project.prepare_dataset(args.data_dir)
        project.visualize_features(args.data_dir)
        model_spoof, history_spoof = project.train_anti_spoofing(train_loader, val_loader)
        metrics = project.evaluate_model(model_spoof, val_loader)
        project.save_config()

        print(f"\nTREINAMENTO COMPLETO!")
        print(f"Modelos salvos em models/")
        print(f"Resultados salvos em results/")

    elif args.mode == 'evaluate':
        print("Modo evaluation...")

    elif args.mode == 'demo':
        print("Modo demo...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPEECH AI SYSTEM - Sistema de Processamento de Voz")
    print("="*70 + "\n")

    main()
