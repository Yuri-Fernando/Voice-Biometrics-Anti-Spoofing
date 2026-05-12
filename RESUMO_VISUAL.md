# Speech AI System - Resumo Visual

## O QUE É?

```
🎙️  ENTRADA          🧠 PROCESSAMENTO         ✅ SAÍDA
┌─────────────┐      ┌──────────────────┐      ┌──────────────┐
│  Arquivo    │      │ 1. Carregar      │      │ REAL         │
│  de Áudio   │─────→│ 2. Extrair       │─────→│ Confiança:   │
│  .wav       │      │    Features      │      │ 97.6%        │
└─────────────┘      │ 3. Rede Neural   │      └──────────────┘
                     │ 4. Classificar   │
                     └──────────────────┘
```

**Função**: Detectar se um áudio é de uma pessoa real ou foi gerado por IA

---

## CASOS DE USO

| Caso | Exemplo |
|------|---------|
| 🔐 **Segurança** | Banco detecta fraude por clonagem de voz |
| 👤 **Biometria** | Smartphone desbloqueia apenas com voz real |
| 🎬 **Forense** | Polícia valida autenticidade de áudio em investigação |
| 📱 **Social Media** | TikTok remove deepfakes de voz antes de viralizarem |
| 🤖 **Anti-Scam** | Operadora bloqueia chamadas com voz falsificada |

---

## O PIPELINE COMPLETO

```
┌─────────────────────────────────────────────────────────┐
│                    GERAÇÃO DE DADOS                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 15 áudios REAIS  │  15 áudios FAKE              │   │
│  │ (200Hz+400Hz)    │  (250Hz+350Hz)              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  PREPARAÇÃO DO DATASET                   │
│  Split 80/20 → 24 treino + 6 validação                 │
│  Padding automático → todos com mesmo tamanho           │
│  Data augmentation → pitch shift, time stretch, noise   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    TREINAMENTO (5 ÉPOCAS)               │
│  Otimizador: Adam (lr=0.001)                            │
│  Loss: CrossEntropyLoss                                 │
│  Arquitetura: CNN (1→32→64→128 canais)                  │
│  Resultado: 100% acurácia ✅                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                       AVALIAÇÃO                          │
│  Accuracy:  100.00%    EER:  0.00%                      │
│  Precision: 100.00%    minDCF: 0.00%                    │
│  Recall:    100.00%    ROC-AUC: 1.0000                 │
│  F1-Score:  100.00%    ⚠️ Dados sintéticos              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    VISUALIZAÇÃO                          │
│  ├─ Mel-spectrogramas (Real vs Fake)                    │
│  └─ Histórico de treinamento (Loss & Accuracy)          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                     PREDIÇÃO                             │
│  Arquivo: data/raw/real_0.wav                           │
│  Predição: REAL                                         │
│  Confiança: 97.6%                                       │
└─────────────────────────────────────────────────────────┘
```

---

## POR QUE AS MÉTRICAS ESTÃO 1.000?

### Dados Sintéticos
```
🎵 REAL:  sine(200Hz) + sine(400Hz)  |  ┌──────────┐
          + ruído leve                 |  │ FÁCIL    │
                                       |  │ SEPARAR  │
🤖 FAKE:  sine(250Hz) + sine(350Hz)   |  │ FREQUÊNCIAS
          (sem ruído)                  |  └──────────┘
```

### Dados Reais (ASVspoof 2019)
```
🎵 REAL:  Vozes humanas variadas      |  ┌──────────┐
          + background noise           |  │ DIFÍCIL  │
          + reverberação               |  │ SEPARAR  │
          + sotaques                   |  │ (deepfake
                                       |  │ modernos
🤖 FAKE:  Síntese neural (HiFi-GAN)   |  │ são muito
          + voice conversion           |  │ realistas)
          + TTS avançado               |  └──────────┘
```

---

## COMPARAÇÃO: DADOS DUMMY vs REAIS

```
╔════════════════╦═══════════════╦════════════════╗
║ MÉTRICA        ║  ESTE PROJETO ║  ASVSPOOF 2019 ║
║                ║  (Dummy Data) ║  (Dados Reais) ║
╠════════════════╬═══════════════╬════════════════╣
║ Acurácia       ║     100%      ║     85-95%     ║
║ ROC-AUC        ║      1.0      ║    0.85-0.95   ║
║ EER            ║      0.0%     ║     5-15%      ║
║ Amostras       ║       30      ║    107,000     ║
║ Realismo       ║      BAIXO    ║      ALTO      ║
║ Overfitting    ║      SIM      ║       NÃO      ║
║ Pronto Prod?   ║      NÃO      ║      SIM       ║
╚════════════════╩═══════════════╩════════════════╝
```

---

## COMO FUNCIONA A CNN?

```
Input: Mel-Spectrogram (Imagem de frequências)

┌─────────────────────────────────────┐
│ Entrada: 128 frequências × 300 tempo│
│ ████████████                        │
│ ██████████                          │
│ ███████████                         │  Visualização:
│ █████████                           │  = padrão
│ ████████████                        │    de ruído
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Conv Block 1: Detecta patterns      │
│ (bordas, texturas básicas)          │
│ 1→32 canais                         │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Conv Block 2: Padrões maiores       │
│ (estruturas, harmônicos)            │
│ 32→64 canais                        │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Conv Block 3: Conceitos de alto nível
│ (é voz real ou síntese?)            │
│ 64→128 canais                       │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Fully Connected:                     │
│ Linear(128→64) + ReLU + Dropout     │
│ Linear(64→2)                        │
└─────────────────────────────────────┘
        ↓
Output: [0.98, 0.02]  ← 98% REAL, 2% FAKE
```

---

## FEATURES EXTRAÍDAS

```
ÁUDIO BRUTO                MEL-SPECTROGRAM         MFCC
[0.023,                    ┌─────────────┐         [2.3,
 -0.156,                   │ █ █ █ █     │         -1.5,
 0.089,                    │ █ █ █ █ █   │  ─────→ 0.8,
 -0.042,    ────────────→  │ █ █ █ █ █ █ │         ...,
 0.201,                    │ █ █ █ █ █ █ │         0.2]
 ...] 
 
Sequência    Imagem 2D das   40 números resumindo
de números  frequências     as características
```

---

## MÉTRICAS EXPLICADAS

### Acurácia
```
Modelo acertou?
Sim = ✅ Acertou
Não = ❌ Errou
Acurácia = (Acertos / Total) × 100%
```

### ROC-AUC
```
0.0 = Pior modelo possível (inverte predições)
0.5 = Modelo aleatório (apenas chuta)
1.0 = Modelo perfeito (sempre acerta) ← ESTE PROJETO
```

### EER (Equal Error Rate)
```
EER = Ponto onde False Acceptance = False Rejection

0% = Perfeito (nunca falha) ← ESTE PROJETO
5-10% = Bom (datasets reais)
20%+ = Ruim
```

### F1-Score
```
Equilibra Precision (quantos fake eram realmente fake?)
         e Recall (quantos fake foram detectados?)

0.0 = Péssimo
1.0 = Perfeito ← ESTE PROJETO
```

---

## ARQUIVOS DO PROJETO

```
projeto/
├── README.md                    ← LEIA PRIMEIRO
├── EXPLICACAO_PROJETO.md        ← Detalhado e didático
├── RESUMO_VISUAL.md             ← Este arquivo
├── DOCUMENTACAO.md              ← Técnico e aprofundado
│
├── pipeline.ipynb               ← Jupyter notebook (execute célula por célula)
├── run_pipeline.py              ← Script Python (execute tudo de uma vez)
│
├── src/                         ← Código-fonte
│   ├── preprocessing.py         ← Carrega e normaliza áudio
│   ├── features.py              ← Extrai MFCC, Mel-spec
│   ├── models.py                ← Arquiteturas CNN/ResNet/LSTM
│   ├── dataset.py               ← AudioDataset com padding
│   ├── training.py              ← Loop de treinamento
│   ├── evaluation.py            ← Métricas profissionais
│   ├── diarization.py           ← Speaker diarization
│   ├── predict.py               ← Inferência em novos áudios
│   └── __init__.py              ← Exports
│
├── data/raw/                    ← Áudios gerados automaticamente
├── models/                      ← Modelos treinados salvos
│   ├── best_model.pt            ← Melhor modelo
│   ├── anti_spoof_model.pt      ← Modelo final
│   └── config.json              ← Configuração salva
│
└── results/                     ← Visualizações
    ├── features_comparison.png  ← Real vs Fake spectrograms
    └── training_history.png     ← Loss e Accuracy
```

---

## COMO USAR

### Opção 1: Jupyter Notebook (Recomendado para Aprender)
```bash
jupyter notebook pipeline.ipynb
# Execute célula por célula, entenda cada passo
```

### Opção 2: Script Python (Rápido)
```bash
python run_pipeline.py
# Executa tudo automaticamente em ~2 minutos
```

### Opção 3: Fazer Predição
```python
from src import AudioInference

inference = AudioInference(model_dir='models')
result = inference.predict_spoofing('seu_audio.wav')

print(f"Predição: {result['prediction']}")
print(f"Confiança: {result['confidence']*100:.1f}%")
```

---

## PRÓXIMOS PASSOS (ROADMAP)

### ✅ Feito
- [x] Pipeline completo com dados dummy
- [x] Modelo treinado (100% acurácia)
- [x] Código testado e sem erros
- [x] Documentação detalhada

### 📋 Fácil (1-2 horas)
- [ ] Integrar ASVspoof 2019 (10k amostras)
- [ ] Treinar por 50 épocas
- [ ] Validar ~85% acurácia realista

### 🎯 Médio (1-2 dias)
- [ ] Data augmentation avançada (SpecAugment)
- [ ] Ensemble de múltiplos modelos
- [ ] Deploy com FastAPI
- [ ] API REST funcional

### 🚀 Difícil (1-2 semanas)
- [ ] Usar ECAPA-TDNN (arquitetura state-of-the-art)
- [ ] Treinar com 100k+ amostras
- [ ] Cross-dataset validation
- [ ] Robustez a transformações

---

## REFERÊNCIAS

| Recurso | URL |
|---------|-----|
| ASVspoof Challenge | https://www.asvspoof.org/ |
| PyTorch Audio | https://pytorch.org/audio/stable/ |
| Librosa (áudio) | https://librosa.org/ |
| Research Paper | https://arxiv.org/abs/1904.05441 |

---

## RESUMO FINAL

**Este projeto mostra:**
- Como construir um sistema anti-deepfake de voz
- Pipeline completo: dados → modelo → predição
- Código production-ready (com dados reais)

**Mas com dados sintéticos:**
- Métricas perfeitas (100%) são esperadas
- Não refletem desempenho em produção
- Use ASVspoof 2019 para validação realista

**Takeaway:**
> Modelos são bons em memorizar dados dummy.
> O desafio real é generalizar para dados novos e complexos.

---

**Desenvolvido com PyTorch 2.0+ | 2024**

