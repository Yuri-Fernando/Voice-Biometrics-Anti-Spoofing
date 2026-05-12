"""Script executavel do Speech AI System"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from scipy.io import wavfile
import json
import torch.nn as nn

from src import AudioProcessor, FeatureExtractor, CNNAntiSpoofing, ModelTrainer, ModelEvaluator, AudioInference

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR = 16000
N_MELS = 128

print(f'Device: {DEVICE}\n')

# ============ 1. Gerar Dados ============
print('1. Gerando dados dummy...')
Path('data/raw').mkdir(parents=True, exist_ok=True)
processor = AudioProcessor(sr=SR)

for i in range(15):
    t = np.linspace(0, 2.0, int(SR * 2.0))
    audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    audio += 0.1 * np.random.randn(len(audio))
    audio = processor.normalize_audio(audio)
    audio = (audio * 32767).astype(np.int16)
    wavfile.write(f'data/raw/real_{i}.wav', SR, audio)

for i in range(15):
    t = np.linspace(0, 2.0, int(SR * 2.0))
    audio = np.sin(2 * np.pi * 250 * t) + 0.3 * np.sin(2 * np.pi * 350 * t)
    audio = processor.normalize_audio(audio)
    audio = (audio * 32767).astype(np.int16)
    wavfile.write(f'data/raw/fake_{i}.wav', SR, audio)

print('   30 arquivos criados\n')

# ============ 2. Dataset com Padding ============
print('2. Preparando dataset...')

class SimpleAudioDataset(Dataset):
    def __init__(self, audio_files, labels, sr, n_mels):
        self.files = audio_files
        self.labels = labels
        self.processor = AudioProcessor(sr=sr)
        self.extractor = FeatureExtractor(sr=sr, n_mels=n_mels)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio, _ = self.processor.load_audio(self.files[idx], sr=SR)
        audio = self.processor.normalize_audio(audio)
        features = self.extractor.extract_mel_spectrogram(audio)
        features = self.extractor.normalize_features(features)
        return torch.FloatTensor(features).unsqueeze(0), torch.LongTensor([self.labels[idx]])[0]

def collate_pad(batch):
    features = [x[0] for x in batch]
    labels = torch.LongTensor([x[1] for x in batch])
    max_len = max(f.shape[2] for f in features)
    
    padded = []
    for f in features:
        pad = max_len - f.shape[2]
        if pad > 0:
            f = torch.nn.functional.pad(f, (0, pad))
        padded.append(f)
    
    return torch.cat(padded, dim=0), labels

audio_files = [str(f) for f in Path('data/raw').glob('*.wav')]
labels = [0 if 'real' in f else 1 for f in audio_files]

dataset = SimpleAudioDataset(audio_files, labels, SR, N_MELS)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_pad)
val_loader = DataLoader(val_set, batch_size=8, collate_fn=collate_pad)

print(f'   {len(audio_files)} arquivos, Train: {train_size}, Val: {val_size}\n')

# ============ 3. Treinar ============
print('3. Treinando modelo...')
model = CNNAntiSpoofing().to(DEVICE)
trainer = ModelTrainer(model, device=DEVICE, lr=0.001)
history = trainer.train(train_loader, val_loader, epochs=5, save_dir='models')
print('   Treinamento completo\n')

# ============ 4. Avaliar ============
print('4. Avaliando...')
criterion = nn.CrossEntropyLoss()
val_loss, val_acc, val_preds, val_labels = trainer.evaluate(val_loader, criterion)

model.eval()
all_scores = []
with torch.no_grad():
    for features, labels in val_loader:
        features = features.to(DEVICE)
        outputs = model(features)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        all_scores.append(probs.cpu().numpy())

all_scores = np.vstack(all_scores)
evaluator = ModelEvaluator()
metrics = evaluator.comprehensive_evaluation(np.array(val_labels), np.array(val_preds), all_scores)

print(f'   Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
evaluator.print_metrics(metrics)

# ============ 5. Visualizar Features ============
print('\n5. Visualizando features...')
extractor = FeatureExtractor(sr=SR, n_mels=N_MELS)
audio_real, _ = processor.load_audio('data/raw/real_0.wav', sr=SR)
audio_fake, _ = processor.load_audio('data/raw/fake_0.wav', sr=SR)

mel_real = extractor.extract_mel_spectrogram(audio_real)
mel_fake = extractor.extract_mel_spectrogram(audio_fake)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(mel_real, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title('Real')
axes[1].imshow(mel_fake, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_title('Fake')
plt.tight_layout()

Path('results').mkdir(exist_ok=True)
plt.savefig('results/features.png', dpi=150, bbox_inches='tight')
print('   Salvo em results/features.png')

# ============ 6. Training History ============
print('\n6. Plotando history...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train')
axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train')
axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/history.png', dpi=150, bbox_inches='tight')
print('   Salvo em results/history.png')

# ============ 7. Predicoes ============
print('\n7. Fazendo predicoes...')
inference = AudioInference(model_dir='models')
result = inference.predict_spoofing('data/raw/real_0.wav')

if 'error' not in result:
    print(f'   Predicao: {result["prediction"].upper()}')
    print(f'   Confianca: {result["confidence"]*100:.1f}%')

# ============ 8. Salvar ============
print('\n8. Salvando modelos...')
Path('models').mkdir(exist_ok=True)
trainer.save_model('models/anti_spoof_model.pt')

config = {'sample_rate': SR, 'n_mels': N_MELS, 'device': str(DEVICE)}
with open('models/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('   Salvos em models/')

# ============ Resumo ============
print('\n' + '='*70)
print('PIPELINE COMPLETO EXECUTADO COM SUCESSO')
print('='*70)
print(f'1. Dataset: 30 arquivos (15 reais, 15 fake)')
print(f'2. Treino: {len(history["train_loss"])} epocas')
print(f'3. Accuracy: {max(history["val_acc"]):.2f}%')
print(f'4. EER: {metrics["eer"]*100:.2f}%')
print(f'5. ROC-AUC: {metrics["roc_auc"]:.4f}')
print('='*70)
