"""Script para rodar pipeline completo do Speech AI System"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.io import wavfile
import json

sys.path.insert(0, str(Path.cwd()))

from src import (
    AudioProcessor,
    FeatureExtractor,
    AudioDataset,
    CNNAntiSpoofing,
    ModelTrainer,
    ModelEvaluator,
    AudioInference,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR = 16000
N_MELS = 128

print(f'Device: {DEVICE}')
print(f'Sample Rate: {SR} Hz')

# ============ 1. Gerar Dados ============
print('\n1. Gerando dados dummy...')
Path('data/raw').mkdir(parents=True, exist_ok=True)
processor = AudioProcessor(sr=SR)

for i in range(15):
    duration = 2.0
    t = np.linspace(0, duration, int(SR * duration))
    audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    audio += 0.1 * np.random.randn(len(audio))
    audio = processor.normalize_audio(audio)
    audio = (audio * 32767).astype(np.int16)
    wavfile.write(f'data/raw/real_{i}.wav', SR, audio)

for i in range(15):
    duration = 2.0
    t = np.linspace(0, duration, int(SR * duration))
    audio = np.sin(2 * np.pi * 250 * t) + 0.3 * np.sin(2 * np.pi * 350 * t)
    audio = processor.normalize_audio(audio)
    audio = (audio * 32767).astype(np.int16)
    wavfile.write(f'data/raw/fake_{i}.wav', SR, audio)

print('   30 arquivos criados em data/raw/')

# ============ 2. Preparar Dataset ============
print('\n2. Preparando dataset...')

def collate_fn_pad(batch):
    features_list = [item[0] for item in batch if item[0] is not None]
    labels_list = [item[1] for item in batch if item[0] is not None]

    if not features_list:
        return None, None

    max_len = max(f.shape[2] for f in features_list)
    padded_features = []

    for f in features_list:
        pad_len = max_len - f.shape[2]
        if pad_len > 0:
            f = torch.nn.functional.pad(f, (0, pad_len))
            padded_features.append(f.squeeze(0))
        else:
            padded_features.append(f.squeeze(0))

    features = torch.stack(padded_features, dim=0)
    features = features.unsqueeze(1)
    labels = torch.LongTensor(labels_list)

    return features, labels

audio_files = []
labels = []

for f in Path('data/raw').glob('*.wav'):
    labels.append(0 if 'real' in f.name else 1)
    audio_files.append(str(f))

print(f'   Total: {len(audio_files)} arquivos')
print(f'   Reais: {sum(1 for l in labels if l == 0)}')
print(f'   Fakes: {sum(1 for l in labels if l == 1)}')

dataset = AudioDataset(
    audio_files=audio_files,
    labels=labels,
    sr=SR,
    n_mels=N_MELS,
    feature_type='mel',
    augmentation=True
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_pad)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn_pad)

print(f'   Train: {len(train_dataset)}, Val: {len(val_dataset)}')

# ============ 3. Treinar ============
print('\n3. Treinando modelo anti-spoofing...')
model = CNNAntiSpoofing().to(DEVICE)
trainer = ModelTrainer(model, device=DEVICE, lr=0.001)

history = trainer.train(
    train_loader,
    val_loader,
    epochs=5,
    save_dir='models'
)

print('   Treinamento completo!')

# ============ 4. Avaliar ============
print('\n4. Avaliando modelo...')
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
val_loss, val_acc, val_preds, val_labels = trainer.evaluate(val_loader, criterion)

print(f'   Validation Loss: {val_loss:.4f}')
print(f'   Validation Accuracy: {val_acc:.2f}%')

model.eval()
all_scores = []

with torch.no_grad():
    for features, labels in val_loader:
        if features is None:
            continue
        features = features.to(DEVICE)
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

print('\n   Métricas:')
evaluator.print_metrics(metrics)

# ============ 5. Visualizar Features ============
print('\n5. Visualizando features...')
extractor = FeatureExtractor(sr=SR, n_mels=N_MELS)

audio_real, _ = processor.load_audio('data/raw/real_0.wav', sr=SR)
audio_fake, _ = processor.load_audio('data/raw/fake_0.wav', sr=SR)

mel_real = extractor.extract_mel_spectrogram(audio_real)
mel_fake = extractor.extract_mel_spectrogram(audio_fake)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Features: Real vs Fake', fontsize=14, fontweight='bold')

axes[0].imshow(mel_real, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title('Mel-Spectrogram (Real)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Frequency')

axes[1].imshow(mel_fake, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_title('Mel-Spectrogram (Fake)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
Path('results').mkdir(exist_ok=True)
plt.savefig('results/features_comparison.png', dpi=300, bbox_inches='tight')
print('   Salvo em results/features_comparison.png')

# ============ 6. Training History ============
print('\n6. Plotando training history...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History', fontsize=14, fontweight='bold')

epochs = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2)
axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train', linewidth=2)
axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
print('   Salvo em results/training_history.png')

# ============ 7. Predicoes ============
print('\n7. Fazendo predicoes...')
inference = AudioInference(model_dir='models')

test_file = 'data/raw/real_0.wav'
result = inference.predict_spoofing(test_file)

if 'error' not in result:
    print(f'   Arquivo: {test_file}')
    print(f'   Predicao: {result["prediction"].upper()}')
    print(f'   Confianca: {result["confidence"]*100:.1f}%')
    print(f'   Prob Real: {result["prob_real"]:.4f}')
    print(f'   Prob Fake: {result["prob_fake"]:.4f}')

# ============ 8. Salvar Modelos ============
print('\n8. Salvando modelos...')
Path('models').mkdir(exist_ok=True)
trainer.save_model('models/anti_spoof_model.pt')

config = {
    'sample_rate': SR,
    'n_mels': N_MELS,
    'n_mfcc': 40,
    'device': str(DEVICE),
    'metrics': metrics
}

with open('models/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('   Modelos salvos em models/')

# ============ Resumo ============
print('\n' + '='*70)
print('PIPELINE COMPLETO EXECUTADO COM SUCESSO')
print('='*70)
print(f'1. Dataset: 30 arquivos (15 reais, 15 fake)')
print(f'2. Treino: {len(history["train_loss"])} epocas')
print(f'3. Accuracy: {max(history["val_acc"]):.2f}%')
print(f'4. EER: {metrics["eer"]*100:.2f}%')
print(f'5. ROC-AUC: {metrics["roc_auc"]:.4f}')
print(f'\nProximos passos:')
print(f'1. Treinar com dados reais (ASVspoof)')
print(f'2. Deploy com: python -m uvicorn api.main:app')
print(f'3. Predicoes com: python src/predict.py --audio audio.wav --task spoof')
print('='*70)
