"""Módulo de Avaliação e Métricas"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from typing import Dict, Tuple


class ModelEvaluator:
    """Classe para avaliar modelos com métricas profissionais"""

    @staticmethod
    def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """
        Calcula EER (Equal Error Rate) - métrica chave em biometria

        EER é o threshold onde FAR (False Accept Rate) = FRR (False Reject Rate)

        Args:
            y_true: Labels verdadeiros
            y_scores: Scores ou probabilidades

        Returns:
            Tupla (EER, threshold)
        """
        fpr, fnr, thresholds = [], [], []

        for threshold in np.linspace(0, 1, 1000):
            predictions = (y_scores >= threshold).astype(int)

            tn = np.sum((predictions == 0) & (y_true == 0))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            tp = np.sum((predictions == 1) & (y_true == 1))

            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

            fpr.append(fpr_val)
            fnr.append(fnr_val)
            thresholds.append(threshold)

        fpr = np.array(fpr)
        fnr = np.array(fnr)
        eer_idx = np.argmin(np.abs(fpr - fnr))

        return fpr[eer_idx], thresholds[eer_idx]

    @staticmethod
    def calculate_minDCF(y_true: np.ndarray, y_scores: np.ndarray,
                        p_target: float = 0.01, c_miss: float = 1.0,
                        c_false_alarm: float = 1.0) -> float:
        """
        Calcula minDCF (Minimum Detection Cost Function)

        Métrica alternativa a EER, mais realista para aplicações reais.

        Args:
            y_true: Labels verdadeiros
            y_scores: Scores
            p_target: Probabilidade a priori do alvo
            c_miss: Custo de miss (não detectar alvo)
            c_false_alarm: Custo de falso alarme

        Returns:
            minDCF
        """
        fpr, fnr, _ = ModelEvaluator._get_error_rates(y_true, y_scores)

        dcf = c_miss * fnr * p_target + c_false_alarm * fpr * (1 - p_target)

        return np.min(dcf)

    @staticmethod
    def _get_error_rates(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple:
        """Calcula FPR e FNR para todos os thresholds"""
        fpr, fnr, thresholds = [], [], []

        for threshold in np.linspace(0, 1, 1000):
            predictions = (y_scores >= threshold).astype(int)

            tn = np.sum((predictions == 0) & (y_true == 0))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            tp = np.sum((predictions == 1) & (y_true == 1))

            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

            fpr.append(fpr_val)
            fnr.append(fnr_val)
            thresholds.append(threshold)

        return np.array(fpr), np.array(fnr), np.array(thresholds)

    @staticmethod
    def comprehensive_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_scores: np.ndarray = None) -> Dict:
        """
        Avaliação completa com todas as métricas

        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            y_scores: Probabilidades (para ROC-AUC e EER)

        Returns:
            Dicionário com todas as métricas
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }

        # ROC-AUC
        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                if y_scores.ndim > 1:
                    roc_auc = float(roc_auc_score(y_true, y_scores[:, 1], average='weighted'))
                else:
                    roc_auc = float(roc_auc_score(y_true, y_scores, average='weighted'))
                metrics['roc_auc'] = roc_auc
            except:
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0

        # EER
        if y_scores is not None:
            if y_scores.ndim > 1:
                scores = y_scores[:, 1]
            else:
                scores = y_scores

            eer, eer_threshold = ModelEvaluator.calculate_eer(y_true, scores)
            metrics['eer'] = float(eer)
            metrics['eer_threshold'] = float(eer_threshold)

            # minDCF
            min_dcf = ModelEvaluator.calculate_minDCF(y_true, scores)
            metrics['min_dcf'] = float(min_dcf)
        else:
            metrics['eer'] = 0.0
            metrics['eer_threshold'] = 0.0
            metrics['min_dcf'] = 0.0

        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = 'Confusion Matrix'):
        """Plota confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)

        classes = ['Real', 'Fake']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        # Adicionar valores nas células
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       color='white' if cm[i, j] > cm.max() / 2 else 'black')

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                      title: str = 'ROC Curve'):
        """Plota curva ROC"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_det_curve(y_true: np.ndarray, y_scores: np.ndarray,
                      title: str = 'DET Curve'):
        """Plota DET curve (Detection Error Tradeoff)"""
        fpr, fnr, _ = ModelEvaluator._get_error_rates(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr * 100, fnr * 100, lw=2, label='DET')

        # EER
        eer, _ = ModelEvaluator.calculate_eer(y_true, y_scores)
        ax.plot([eer * 100, eer * 100], [0, 100], 'r--', lw=2, label=f'EER = {eer*100:.2f}%')

        ax.set_xlabel('False Positive Rate (%)')
        ax.set_ylabel('False Negative Rate (%)')
        ax.set_title(title)
        ax.set_xlim([0.01, 100])
        ax.set_ylim([0.01, 100])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_metrics(metrics: Dict, title: str = 'Model Metrics'):
        """Plota resumo de métricas"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Accuracy
        axes[0, 0].bar(['Accuracy'], [metrics.get('accuracy', 0)], color='skyblue')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_ylabel('Score')

        # ROC-AUC
        axes[0, 1].bar(['ROC-AUC'], [metrics.get('roc_auc', 0)], color='lightgreen')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_title('ROC-AUC')
        axes[0, 1].set_ylabel('Score')

        # EER
        axes[1, 0].bar(['EER'], [metrics.get('eer', 0)], color='lightcoral')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].set_title('Equal Error Rate (EER)')
        axes[1, 0].set_ylabel('Error Rate')

        # F1-Score
        axes[1, 1].bar(['F1-Score'], [metrics.get('f1', 0)], color='plum')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].set_title('F1-Score')
        axes[1, 1].set_ylabel('Score')

        plt.tight_layout()
        return fig

    @staticmethod
    def print_metrics(metrics: Dict):
        """Printa métricas de forma legível"""
        print("\n" + "="*50)
        print("[INFO] MÉTRICAS DE AVALIAÇÃO")
        print("="*50)
        print(f"Accuracy:     {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
        print(f"Precision:    {metrics.get('precision', 0):.4f}")
        print(f"Recall:       {metrics.get('recall', 0):.4f}")
        print(f"F1-Score:     {metrics.get('f1', 0):.4f}")
        print(f"ROC-AUC:      {metrics.get('roc_auc', 0):.4f}")
        print(f"EER:          {metrics.get('eer', 0):.4f} ({metrics.get('eer', 0)*100:.2f}%)")
        print(f"minDCF:       {metrics.get('min_dcf', 0):.4f}")
        print("="*50 + "\n")


if __name__ == "__main__":
    print("[OK] Módulo evaluation.py importado com sucesso!")
