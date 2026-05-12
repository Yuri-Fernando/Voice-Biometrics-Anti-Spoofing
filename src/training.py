"""Módulo de Treinamento de Modelos"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, List
import json
from pathlib import Path


class ModelTrainer:
    """Classe para treinar modelos de áudio"""

    def __init__(self, model: nn.Module, device: torch.device = None, lr: float = 0.001):
        """
        Inicializa trainer

        Args:
            model: Modelo PyTorch
            device: Device (cuda/cpu)
            lr: Learning rate
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        self.best_val_acc = 0.0

    def train_epoch(self, dataloader: DataLoader, criterion,
                    optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Treina uma época

        Args:
            dataloader: DataLoader de treinamento
            criterion: Função de loss
            optimizer: Otimizador

        Returns:
            Tupla (loss_médio, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc='Training')

        for features, labels in progress_bar:
            if features is None:
                continue

            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Métricas
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0

        return avg_loss, accuracy

    def evaluate(self, dataloader: DataLoader, criterion) -> Tuple[float, float, List, List]:
        """
        Avalia o modelo

        Args:
            dataloader: DataLoader de validação
            criterion: Função de loss

        Returns:
            Tupla (loss_médio, accuracy, predictions, labels)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc='Evaluating'):
                if features is None:
                    continue

                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0

        return avg_loss, accuracy, all_preds, all_labels

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 10, save_dir: str = 'models',
              early_stopping: int = 5) -> Dict:
        """
        Loop principal de treinamento

        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação
            epochs: Número de épocas
            save_dir: Diretório para salvar modelos
            early_stopping: Parar se val accuracy não melhorar em N épocas

        Returns:
            Histórico de treinamento
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        epochs_no_improve = 0

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")

            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Validation
            val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)

            # Armazenar histórico
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            # Learning rate scheduling
            scheduler.step(val_acc)

            # Salvar melhor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(f"{save_dir}/best_model.pt")
                epochs_no_improve = 0
                print(f"  [BEST] Melhor modelo salvo! (Acc: {val_acc:.2f}%)")
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= early_stopping:
                print(f"\n[EARLY STOP] Early stopping ativado apos {early_stopping} epocas sem melhora")
                break

        return self.history

    def save_model(self, path: str):
        """
        Salva o modelo

        Args:
            path: Caminho para salvar
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[OK] Modelo salvo em {path}")

    def load_model(self, path: str):
        """
        Carrega o modelo

        Args:
            path: Caminho do modelo
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[OK] Modelo carregado de {path}")

    def save_checkpoint(self, path: str, optimizer_state: Dict = None):
        """
        Salva checkpoint completo (modelo + otimizador)

        Args:
            path: Caminho para salvar
            optimizer_state: Estado do otimizador
        """
        checkpoint = {
            'model_state': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }

        if optimizer_state:
            checkpoint['optimizer_state'] = optimizer_state

        torch.save(checkpoint, path)
        print(f"[OK] Checkpoint salvo em {path}")

    def load_checkpoint(self, path: str):
        """
        Carrega checkpoint

        Args:
            path: Caminho do checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', self.history)
        print(f"[OK] Checkpoint carregado de {path}")

    def get_summary(self) -> str:
        """Retorna resumo do treinamento"""
        summary = "Treinamento Summary\n"
        summary += f"Melhor Accuracy: {self.best_val_acc:.2f}%\n"
        summary += f"Épocas completadas: {len(self.history['train_loss'])}\n"
        return summary


if __name__ == "__main__":
    print("[OK] Modulo training.py importado com sucesso!")
