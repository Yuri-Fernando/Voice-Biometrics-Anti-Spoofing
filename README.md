# Speech AI System — Voice Biometrics & Anti-Spoofing

Sistema de Inteligência Artificial para **processamento de áudio, detecção de spoofing de voz e análise biométrica de locutores**, desenvolvido com Deep Learning em PyTorch.

A aplicação combina processamento de sinais de áudio, extração de características acústicas, redes neurais convolucionais, embeddings de voz, speaker verification e diarização.

---

## Status

🟢 **Concluído — Projeto de portfólio / Speech AI & Voice Biometrics**

O pipeline principal está implementado e funcional, contemplando:

* Processamento e normalização de áudio;
* Extração de features acústicas;
* Classificação anti-spoofing;
* Speaker Verification;
* Speaker Diarization;
* Treinamento e inferência com PyTorch;
* Data augmentation;
* Avaliação por métricas de classificação e biometria;
* API REST com FastAPI;
* Notebook demonstrativo;
* Scripts independentes de treinamento e inferência;
* Estrutura modular para evolução;
* Documentação técnica.

A versão concluída utiliza um **dataset sintético controlado**, destinado à demonstração e validação do pipeline. A utilização de datasets reais especializados, como ASVspoof 2019, permanece como próxima etapa de validação experimental.

---

## Sobre o projeto

O sistema responde a duas perguntas principais:

1. **O áudio é real ou falsificado?**
2. **Quem é o locutor?**

A arquitetura também possui uma camada de **speaker diarization**, permitindo segmentar áudios com múltiplos locutores e identificar mudanças entre falantes.

O projeto explora aplicações de IA relacionadas a **Voice Biometrics, Anti-Spoofing, Audio Deepfake Detection e Speaker Analysis**.

---

## Funcionalidades

### Anti-Spoofing

Classificação de áudio como:

* `REAL`
* `FAKE`

A entrada é um arquivo de áudio e a saída inclui:

* Predição;
* Confiança;
* Probabilidade por classe.

### Speaker Verification

O sistema extrai **embeddings de voz de 256 dimensões**, permitindo:

* Comparação entre vozes;
* Verificação de similaridade;
* Identificação de locutor;
* Experimentação com autenticação biométrica.

### Speaker Diarization

Processamento de áudios com múltiplos locutores para:

* Identificar mudanças de falante;
* Determinar quem falou em cada intervalo;
* Gerar segmentos por locutor.

### Processamento de áudio

* Carregamento de arquivos;
* Normalização;
* Pitch shifting;
* Time stretching;
* Adição de ruído;
* Remoção de silêncio;
* Voice Activity Detection simplificada.

---

## Arquitetura

```text
                     ┌─────────────────────┐
                     │     Audio Input     │
                     │ WAV / MP3 / Audio   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Audio Preprocessing │
                     │ Normalize / VAD     │
                     │ Augmentation        │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Feature Extraction  │
                     │ Mel / MFCC / Spec   │
                     └──────────┬──────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌──────────────────┐    ┌──────────────────┐
          │ Anti-Spoof Model │    │ Speaker Model    │
          │ CNN / ResNet     │    │ Embeddings       │
          │ LSTM             │    │ 256 dimensions   │
          └────────┬─────────┘    └────────┬─────────┘
                   │                       │
                   ▼                       ▼
          ┌──────────────────┐    ┌──────────────────┐
          │ Real / Fake      │    │ Speaker Analysis │
          │ + Confidence     │    │ / Verification   │
          └──────────────────┘    └──────────────────┘
                   │                       │
                   └───────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ API / Python /       │
                    │ Notebook / Inference │
                    └──────────────────────┘
```

---

## Pipeline de Dados

O fluxo completo de processamento é:

```text
1. Carregamento do áudio
        ↓
2. Normalização
        ↓
3. Data Augmentation (opcional)
        ↓
4. Extração de features
        ↓
5. Normalização das features
        ↓
6. Construção do Dataset
        ↓
7. Batch / DataLoader
        ↓
8. Forward Pass
        ↓
9. Loss + Otimização
        ↓
10. Validação
        ↓
11. Métricas
        ↓
12. Salvamento do modelo
        ↓
13. Inferência via Python ou API
```

Essa sequência está implementada entre os módulos de preprocessing, features, dataset, training, evaluation e inference.

---

## Tecnologias utilizadas

### Linguagem

* Python 3.10+

### Deep Learning

* PyTorch 2.0+
* CNN
* ResNet
* LSTM

### Processamento de áudio

* librosa
* NumPy
* SciPy

### Machine Learning

* scikit-learn

### Dados

* Pandas

### Visualização

* Matplotlib

### API

* FastAPI
* Uvicorn

### Desenvolvimento

* Jupyter Notebook
* CUDA opcional

---

## Dataset

### Dataset atual

A versão atual utiliza **dados completamente sintéticos**:

| Classe       | Geração                     |
| ------------ | --------------------------- |
| Real         | Senoides de 200 Hz + 400 Hz |
| Fake         | Senoides de 250 Hz + 350 Hz |
| Total        | 30 áudios                   |
| Distribuição | 15 real / 15 fake           |

### Por que utilizar dados sintéticos?

O dataset foi utilizado para validar a implementação do pipeline, incluindo:

* carregamento;
* preparação;
* treinamento;
* classificação;
* avaliação;
* inferência.

Entretanto, a separação entre as classes é artificial e não representa a complexidade de áudio de voz real.

---

## Resultados Atuais

Com o dataset sintético, o sistema apresenta:

| Métrica   | Resultado |
| --------- | --------: |
| Accuracy  |      100% |
| ROC-AUC   |       1.0 |
| EER       |      0.0% |
| Precision |      100% |
| Recall    |      100% |
| F1-Score  |       1.0 |

Esses resultados são válidos **somente para o dataset sintético utilizado** e não representam desempenho esperado em dados reais.

### Interpretação

As métricas perfeitas decorrem principalmente de:

* separação espectral simples entre classes;
* ausência de ruído real;
* ausência de reverberação;
* ausência de compressão;
* dataset extremamente pequeno;
* padrões de frequência fixos.

---

## Extração de Features

O sistema possui um módulo dedicado à extração de características acústicas.

### Features implementadas

* MFCC — 40 coeficientes;
* Mel-spectrogram — 128 bins;
* Spectrogram;
* Delta MFCC;
* Chroma;
* Zero Crossing Rate;
* Spectral Centroid;
* Spectral Rolloff;
* RMS Energy.

Exemplo:

```python
from src import FeatureExtractor

extractor = FeatureExtractor(
    sr=16000,
    n_mels=128
)

mel_spec = extractor.extract_mel_spectrogram(audio)
mfcc = extractor.extract_mfcc(audio)

features = extractor.extract_combined_features(
    audio,
    ['mel', 'mfcc']
)
```

---

## Modelos

### CNNAntiSpoofing

Modelo principal para classificação de áudio real/fake:

```text
Input
(1, 128, T)
    ↓
Conv1D(1, 32)
    ↓
ReLU + MaxPool + BatchNorm
    ↓
Conv1D(32, 64)
    ↓
ReLU + MaxPool + BatchNorm
    ↓
Conv1D(64, 128)
    ↓
ReLU + MaxPool + BatchNorm
    ↓
AdaptiveAvgPool
    ↓
Linear(128, 64)
    ↓
ReLU + Dropout
    ↓
Linear(64, 2)
    ↓
REAL / FAKE
```

### SpeakerEmbedding

Modelo para geração de embeddings:

```text
Input
(1, 128, T)
    ↓
Convolutional Blocks
    ↓
AdaptiveAvgPool
    ↓
Linear(128, 256)
    ↓
L2 Normalize
    ↓
256-dimensional embedding
```

### Modelos disponíveis

* `CNNAntiSpoofing`
* `SpeakerEmbedding`
* `ResNetAntiSpoofing`
* `LSTMAntiSpoofing`
* `SpeechAISystem`

---

## Treinamento

O `ModelTrainer` implementa o ciclo completo de treinamento.

### Recursos

* Early stopping;
* Learning rate scheduling;
* Gradient clipping;
* Checkpoint do melhor modelo;
* Validação a cada época.

Exemplo:

```python
from src import CNNAntiSpoofing, ModelTrainer

model = CNNAntiSpoofing().to(device)

trainer = ModelTrainer(
    model,
    device=device,
    lr=0.001
)

history = trainer.train(
    train_loader,
    val_loader,
    epochs=10,
    save_dir='models'
)

trainer.save_model('models/model.pt')
```

---

## Avaliação

O módulo `ModelEvaluator` calcula métricas utilizadas na avaliação de classificadores e sistemas biométricos:

* Accuracy;
* Precision;
* Recall;
* F1-Score;
* ROC-AUC;
* EER;
* minDCF;
* Confusion Matrix.

O **EER (Equal Error Rate)** é particularmente relevante em sistemas biométricos, pois representa o ponto em que FAR e FRR se igualam.

---

## Speaker Diarization

O módulo `SpeakerDiarization` segmenta áudios com múltiplos locutores.

Exemplo:

```python
from src import SpeakerDiarization

diarizer = SpeakerDiarization(
    n_speakers=2
)

result = diarizer.diarize(
    'audio.wav',
    model
)
```

O resultado contém informações como:

```text
start_time
end_time
speaker_id
duration
```

---

## Inferência

A classe `AudioInference` permite carregar modelos treinados e realizar predições.

```python
from src import AudioInference

inference = AudioInference(
    model_dir='models'
)

result = inference.predict_spoofing(
    'audio.wav'
)

print(result['prediction'])
print(result['confidence'])
```

---

## API REST

A aplicação disponibiliza uma API utilizando FastAPI.

Inicialização:

```bash
python -m uvicorn api.main:app --reload
```

Documentação Swagger:

```text
http://localhost:8000/docs
```

---

### GET `/`

Retorna informações do servidor.

### GET `/health`

Health check da aplicação.

### POST `/predict/spoof`

Detecta spoofing no áudio enviado.

```bash
curl -X POST http://localhost:8000/predict/spoof \
  -F "file=@audio.wav"
```

Exemplo de resposta:

```json
{
  "prediction": "real",
  "confidence": 0.92,
  "prob_real": 0.92,
  "prob_fake": 0.08
}
```

### POST `/predict/verify`

Realiza análise de speaker verification.

```bash
curl -X POST http://localhost:8000/predict/verify \
  -F "file=@audio.wav"
```

### POST `/predict/diarize`

Executa diarização.

```bash
curl -X POST http://localhost:8000/predict/diarize \
  -F "file=@audio.wav" \
  -F "n_speakers=2"
```

### POST `/extract-features`

Extrai features acústicas.

```bash
curl -X POST http://localhost:8000/extract-features \
  -F "file=@audio.wav" \
  -F "feature_type=mel"
```

---

## Como Executar

### Requisitos

* Python 3.10+
* PyTorch 2.0+
* librosa
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* FastAPI
* Uvicorn
* Jupyter

GPU é opcional:

* CUDA 11.8+
* Aproximadamente 2 GB de VRAM por modelo.

### Instalação

```bash
pip install -r requirements.txt
```

---

### Opção 1 — Notebook

```bash
jupyter notebook notebook_demo.ipynb
```

O notebook possui 10 células que percorrem:

1. Geração dos dados;
2. Preparação;
3. Treinamento;
4. Avaliação;
5. Visualização;
6. Predição.

---

### Opção 2 — Treinamento via CLI

```bash
python src/main.py --mode train --epochs 5
```

### Predição

```bash
python src/predict.py \
  --audio audio.wav \
  --task spoof
```

### Modelo específico

```bash
python src/train.py \
  --model-type anti_spoof \
  --epochs 20
```

---

### Opção 3 — API

```bash
python -m uvicorn api.main:app --reload
```

---

## Configuração

O arquivo `config.json` utiliza parâmetros como:

```json
{
  "sample_rate": 16000,
  "n_mfcc": 40,
  "n_mels": 128,
  "batch_size": 32,
  "learning_rate": 0.001,
  "epochs": 10,
  "feature_type": "mel",
  "test_split": 0.2,
  "val_split": 0.1
}
```

---

## Estrutura do Projeto

```text
Speech AI System/
│
├── src/
│   ├── main.py
│   ├── train.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── dataset.py
│   ├── training.py
│   ├── evaluation.py
│   ├── diarization.py
│   └── __init__.py
│
├── api/
│   └── main.py
│
├── notebook_demo.ipynb
├── DOCUMENTACAO.md
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│
├── models/
│
└── results/
```

---

## Componentes Principais

### `AudioProcessor`

Responsável por:

* carregamento;
* normalização;
* pitch shifting;
* time stretching;
* adição de ruído;
* remoção de silêncio.

### `FeatureExtractor`

Responsável pela extração das características acústicas.

### `ModelTrainer`

Responsável pelo treinamento, validação, checkpoint e otimização.

### `ModelEvaluator`

Responsável pelas métricas de classificação e biometria.

### `AudioInference`

Responsável pela inferência dos modelos treinados.

### `SpeakerDiarization`

Responsável pela segmentação de múltiplos locutores.

---

## Casos de Uso

O projeto pode servir como base experimental para:

1. **Autenticação biométrica por voz**
2. **Detecção de áudio sintetizado**
3. **Análise de segurança em sistemas de voz**
4. **Speaker verification**
5. **Speaker diarization**
6. **Análise forense de áudio**
7. **Pesquisa em detecção de deepfakes de voz**

Esses cenários representam possibilidades de aplicação do pipeline, e não validações comerciais ou clínicas do sistema.

---

## Limitações Atuais

### Dataset

A principal limitação é o conjunto sintético utilizado na versão atual:

* Apenas 30 amostras;
* Frequências artificiais e fixas;
* Ausência de ruído real;
* Ausência de reverberação;
* Ausência de compressão;
* Variabilidade limitada;
* Forte risco de overfitting aos padrões artificiais.

Portanto, os resultados de 100% não devem ser interpretados como desempenho de um detector de deepfake em condições reais.

### Sistema

* Sample rate esperado de 16 kHz;
* Melhor comportamento com áudios de aproximadamente 2–10 segundos;
* Diarização depende do número de locutores informado;
* Cálculo do EER pode ser computacionalmente intensivo.

---

## Dataset Real — Próxima Validação

Para avançar da demonstração para uma avaliação mais representativa, o projeto pode ser executado com bases especializadas como:

### ASVspoof 2019

Dataset direcionado à avaliação de sistemas de **Automatic Speaker Verification e anti-spoofing**.

Outras bases de referência:

* VoxCeleb;
* LibriSpeech;
* Common Voice.

---

## Próximas Melhorias

* Fine-tuning com modelos pré-treinados;
* Transfer Learning;
* Attention mechanisms;
* Ensemble de modelos;
* TorchScript;
* Quantização;
* Processamento de áudio em streaming;
* Testes unitários;
* Containerização com Docker;
* Avaliação com datasets reais;
* Melhorias na generalização do anti-spoofing.

---

## Referências Científicas

* **ASVspoof Challenge** — https://arxiv.org/abs/1904.05441
* **ECAPA-TDNN** — https://arxiv.org/abs/2005.07143
* **VoxCeleb** — https://arxiv.org/abs/1706.08612
* **PyTorch Audio** — https://pytorch.org/audio/

---

## O que este projeto demonstra

* Deep Learning aplicado a áudio;
* Processamento digital de sinais;
* Extração de características acústicas;
* CNNs, ResNet e LSTM;
* Voice Biometrics;
* Anti-Spoofing;
* Speaker Verification;
* Speaker Diarization;
* Avaliação com métricas biométricas;
* Desenvolvimento de API REST;
* Organização modular de projetos de ML;
* Treinamento e inferência em CPU/GPU;
* Integração entre Machine Learning e software.

---

## Licença

MIT License.

---

## Autor

**Yuri Fernando Dubbern**

AI/ML Engineer · Generative AI · Machine Learning · Data Science · Signal Processing

[LinkedIn](https://www.linkedin.com/in/yuridubbern) · [GitHub](https://github.com/Yuri-Fernando) · [Lattes](http://lattes.cnpq.br/7151392692642166) · [Linktree](https://linktr.ee/yuri.f.dubbern)

---

## Observação Final

O projeto está **concluído em sua versão atual**, com pipeline funcional e documentação técnica. A próxima etapa não é completar a arquitetura básica, mas **avaliar a generalização do sistema com datasets reais de voz e ataques de spoofing**, especialmente ASVspoof 2019.

Essa distinção é importante: o projeto já demonstra engenharia de **Speech AI + Deep Learning + Biometrics + API**, enquanto a validação em dados reais representa uma evolução experimental natural.
