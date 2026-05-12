"""Script de treinamento standalone para modelos de voz"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from src import (
    AudioDataset, CNNAntiSpoofing, SpeakerEmbedding,
    ModelTrainer, ModelEvaluator
)


def train_model(config: dict):
    """Treina modelo com configuração"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"🎓 TREINAMENTO DE MODELO")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}\n")

    # Coletar arquivos
    data_dir = Path(config['data_dir'])
    audio_files = []
    labels = []

    for audio_file in data_dir.glob('*.wav'):
        if 'real' in audio_file.name:
            labels.append(0)
        elif 'fake' in audio_file.name:
            labels.append(1)
        else:
            continue
        audio_files.append(str(audio_file))

    if len(audio_files) == 0:
        print("[ERROR] Nenhum arquivo de áudio encontrado!")
        return

    print(f"[INFO] Dataset: {len(audio_files)} arquivos")
    print(f"   - {sum(1 for l in labels if l == 0)} reais")
    print(f"   - {sum(1 for l in labels if l == 1)} fakes\n")

    # Dataset
    dataset = AudioDataset(
        audio_files=audio_files,
        labels=labels,
        sr=config['sample_rate'],
        n_mels=config['n_mels'],
        feature_type=config['feature_type'],
        augmentation=config.get('augmentation', True)
    )

    # Split
    train_size = int((1 - config['val_split']) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size']
    )

    # Modelo
    if config['model_type'] == 'anti_spoof':
        model = CNNAntiSpoofing().to(device)
    else:
        model = SpeakerEmbedding(embedding_dim=config.get('embedding_dim', 256)).to(device)

    print(f"🧠 Modelo: {config['model_type']}")
    print(f"   Parâmetros: {sum(p.numel() for p in model.parameters()):,}\n")

    # Trainer
    trainer = ModelTrainer(model, device=device, lr=config['learning_rate'])

    # Treinar
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=config['epochs'],
        save_dir=config['model_dir'],
        early_stopping=config.get('early_stopping', 5)
    )

    # Avaliar
    print(f"\n📈 Avaliando no validation set...")
    val_loss, val_acc, val_preds, val_labels = trainer.evaluate(
        val_loader, torch.nn.CrossEntropyLoss()
    )

    # Scores
    all_scores = []
    model.eval()
    with torch.no_grad():
        for features, labels in val_loader:
            if features is None:
                continue
            features = features.to(device)
            outputs = model(features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_scores.append(probs.cpu().numpy())

    all_scores = np.vstack(all_scores)

    # Métricas
    evaluator = ModelEvaluator()
    metrics = evaluator.comprehensive_evaluation(
        np.array(val_labels),
        np.array(val_preds),
        all_scores
    )

    evaluator.print_metrics(metrics)

    # Salvar modelo
    model_path = Path(config['model_dir']) / f"{config['model_type']}_model.pt"
    trainer.save_model(str(model_path))

    # Salvar métricas
    metrics_path = Path(config['model_dir']) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] TREINAMENTO COMPLETO!")
    print(f"   Modelo: {model_path}")
    print(f"   Métricas: {metrics_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinar modelo de voz')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Arquivo de configuração')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Diretório com dados')
    parser.add_argument('--model-type', type=str, default='anti_spoof',
                       choices=['anti_spoof', 'speaker_embed'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    # Carregar ou criar configuração
    if Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            'sample_rate': 16000,
            'n_mfcc': 40,
            'n_mels': 128,
            'data_dir': args.data_dir,
            'model_dir': 'models',
            'model_type': args.model_type,
            'batch_size': args.batch_size,
            'learning_rate': 0.001,
            'epochs': args.epochs,
            'val_split': 0.2,
            'feature_type': 'mel',
            'augmentation': True,
        }

    train_model(config)
