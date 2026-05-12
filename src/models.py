"""Módulo de Modelos Deep Learning com PyTorch"""

import torch
import torch.nn as nn
from typing import Tuple


class CNNAntiSpoofing(nn.Module):
    """CNN para classificação de spoofing (Real vs Fake)"""

    def __init__(self, input_channels: int = 1, num_classes: int = 2):
        """
        Inicializa modelo CNN anti-spoofing

        Args:
            input_channels: Número de canais de entrada (1 = mono)
            num_classes: Número de classes (2 = Real/Fake)
        """
        super(CNNAntiSpoofing, self).__init__()

        # Bloco 1: Conv + ReLU + MaxPool + BatchNorm
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )

        # Bloco 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )

        # Bloco 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128)
        )

        # Pooling adaptativo e camadas fully connected
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Logits para cada classe
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class SpeakerEmbedding(nn.Module):
    """Rede para extrair embeddings de voz (Speaker Vectors)"""

    def __init__(self, embedding_dim: int = 256, input_channels: int = 1):
        """
        Inicializa modelo de embeddings

        Args:
            embedding_dim: Dimensão do vetor de embedding
            input_channels: Número de canais de entrada
        """
        super(SpeakerEmbedding, self).__init__()

        # Blocos convolucionais
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embed = nn.Linear(128, embedding_dim)

    def forward(self, x):
        """
        Forward pass que retorna embeddings

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Embeddings normalizados (batch, embedding_dim)
        """
        # Conv block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Conv block 2
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Conv block 3
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # FC para embeddings
        embedding = self.fc_embed(x)

        # L2 normalize (importante para similaridade coseno)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding


class SpeechAISystem(nn.Module):
    """Sistema combinado: Anti-spoofing + Speaker Verification"""

    def __init__(self, embedding_dim: int = 256):
        """
        Inicializa sistema completo

        Args:
            embedding_dim: Dimensão dos embeddings de voz
        """
        super(SpeechAISystem, self).__init__()

        self.anti_spoof_net = CNNAntiSpoofing(num_classes=2)
        self.speaker_embed_net = SpeakerEmbedding(embedding_dim=embedding_dim)

    def forward(self, x, task: str = 'both') -> Tuple:
        """
        Forward pass do sistema

        Args:
            x: Input tensor (batch, 1, height, width)
            task: Qual tarefa executar ('spoof', 'speaker', ou 'both')

        Returns:
            Dependendo da tarefa:
            - 'spoof': logits de classificação (batch, 2)
            - 'speaker': embeddings (batch, embedding_dim)
            - 'both': tupla (spoof_logits, speaker_embeddings)
        """
        if task == 'spoof':
            return self.anti_spoof_net(x)
        elif task == 'speaker':
            return self.speaker_embed_net(x)
        else:  # both
            spoof_pred = self.anti_spoof_net(x)
            speaker_emb = self.speaker_embed_net(x)
            return spoof_pred, speaker_emb


class ResNetBlock(nn.Module):
    """Bloco residual simples"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetAntiSpoofing(nn.Module):
    """ResNet-style architecture para anti-spoofing"""

    def __init__(self, num_classes: int = 2):
        super(ResNetAntiSpoofing, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Blocos residuais
        self.layer1 = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),
        )

        self.layer2 = nn.Sequential(
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128),
        )

        self.layer3 = nn.Sequential(
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class LSTMAntiSpoofing(nn.Module):
    """LSTM-based architecture para anti-spoofing"""

    def __init__(self, input_size: int = 128, hidden_size: int = 256,
                 num_layers: int = 2, num_classes: int = 2):
        super(LSTMAntiSpoofing, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        # Reshape para LSTM: (batch, time_steps, features)
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(2), -1)  # (batch, height, width*channels)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Usar últim hidden state
        x = h_n[-1]  # (batch, hidden_size)

        # FC layers
        x = self.fc_layers(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    Conta número total de parâmetros treináveis do modelo

    Args:
        model: Modelo PyTorch

    Returns:
        Número de parâmetros
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: Tuple) -> str:
    """
    Retorna resumo do modelo

    Args:
        model: Modelo PyTorch
        input_shape: Formato de entrada (ex: (1, 128, 87))

    Returns:
        String com sumário
    """
    summary = f"Model: {model.__class__.__name__}\n"
    summary += f"Input shape: {input_shape}\n"
    summary += f"Total parameters: {count_parameters(model):,}\n"
    summary += f"Trainable parameters: {count_parameters(model):,}\n"
    return summary


if __name__ == "__main__":
    # Teste dos modelos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CNN Anti-spoofing
    model_cnn = CNNAntiSpoofing().to(device)
    print(f"[OK] CNNAntiSpoofing - Parâmetros: {count_parameters(model_cnn):,}")

    # Speaker Embedding
    model_embed = SpeakerEmbedding(embedding_dim=256).to(device)
    print(f"[OK] SpeakerEmbedding - Parâmetros: {count_parameters(model_embed):,}")

    # Sistema combinado
    model_system = SpeechAISystem(embedding_dim=256).to(device)
    print(f"[OK] SpeechAISystem - Parâmetros: {count_parameters(model_system):,}")

    # Teste forward pass
    x = torch.randn(4, 1, 128, 87).to(device)  # Batch de 4

    spoof_out = model_cnn(x)
    print(f"[OK] CNN output shape: {spoof_out.shape}")

    embed_out = model_embed(x)
    print(f"[OK] Embedding output shape: {embed_out.shape}")

    spoof_out, embed_out = model_system(x, task='both')
    print(f"[OK] System spoof output: {spoof_out.shape}, embedding: {embed_out.shape}")
