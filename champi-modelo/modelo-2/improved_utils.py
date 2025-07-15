import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def collate_fn(batch):
    """Mejorado para manejar mejor las secuencias de diferentes longitudes"""
    clips = [item['clip'] for item in batch]
    songs = [item['song'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    clip_ids = [item['clip_id'] for item in batch]
    valid = torch.tensor([item['valid'] for item in batch])

    return {
        'clip': clips,
        'song': songs,
        'target': targets,
        'clip_id': clip_ids,
        'valid': valid,
    }

def improved_loss_fn(ts_pred, conf_pred, ts_true, delta=0.05, conf_weight=0.3):
    """
    Función de pérdida mejorada que considera:
    - Error temporal con diferentes escalas
    - Confianza calibrada según el error real
    - Balanceo entre ambas pérdidas
    """
    # Pérdida temporal con Huber loss (más robusta que MSE)
    temporal_error = torch.abs(ts_pred - ts_true)
    ts_loss = F.huber_loss(ts_pred, ts_true, delta=delta)
    
    # Confianza objetivo: alta si error < delta, baja si error > delta*2
    success_threshold = delta
    failure_threshold = delta * 3
    
    # Confianza objetivo suave (no binaria)
    conf_target = torch.clamp(
        1.0 - (temporal_error - success_threshold) / (failure_threshold - success_threshold),
        0.0, 1.0
    )
    
    # Pérdida de confianza
    conf_loss = F.mse_loss(conf_pred, conf_target)
    
    # Pérdida adicional: penalizar alta confianza con error alto
    overconfidence_penalty = torch.mean(
        torch.relu(conf_pred - 0.5) * torch.relu(temporal_error - delta)
    )
    
    # Combinar pérdidas
    total_loss = ts_loss + conf_weight * conf_loss + 0.1 * overconfidence_penalty
    
    return total_loss, {
        'ts_loss': ts_loss.item(),
        'conf_loss': conf_loss.item(), 
        'overconf_penalty': overconfidence_penalty.item(),
        'mean_error': temporal_error.mean().item(),
        'mean_confidence': conf_pred.mean().item()
    }

def focal_loss(ts_pred, conf_pred, ts_true, alpha=0.25, gamma=2.0, delta=0.05):
    """
    Focal loss alternativo para casos donde hay muchos ejemplos fáciles/difíciles
    """
    temporal_error = torch.abs(ts_pred - ts_true)
    
    # Pérdida temporal estándar
    ts_loss = F.smooth_l1_loss(ts_pred, ts_true)
    
    # Focal loss para confianza
    success = (temporal_error < delta).float()
    
    # Cross entropy con focal weighting
    ce_loss = F.binary_cross_entropy(conf_pred, success, reduction='none')
    p_t = success * conf_pred + (1 - success) * (1 - conf_pred)
    focal_weight = alpha * torch.pow(1 - p_t, gamma)
    conf_loss = torch.mean(focal_weight * ce_loss)
    
    return ts_loss + conf_loss, {
        'ts_loss': ts_loss.item(),
        'conf_loss': conf_loss.item()
    }

class TemporalAugmenter:
    """Aumenta datos temporalmente para mejorar robustez"""
    
    def __init__(self, noise_std=0.01, time_shift_ratio=0.1):
        self.noise_std = noise_std
        self.time_shift_ratio = time_shift_ratio
    
    def augment_embeddings(self, embeddings, apply_noise=True, apply_dropout=True):
        """
        Aplica aumentos a los embeddings
        embeddings: [T, 512]
        """
        if apply_noise:
            noise = torch.randn_like(embeddings) * self.noise_std
            embeddings = embeddings + noise
        
        if apply_dropout and torch.rand(1) < 0.3:
            # Dropout temporal: eliminar algunos frames aleatorios
            keep_ratio = 0.8 + torch.rand(1) * 0.15  # Entre 80-95%
            keep_frames = int(embeddings.shape[0] * keep_ratio)
            indices = torch.randperm(embeddings.shape[0])[:keep_frames]
            indices = torch.sort(indices)[0]
            embeddings = embeddings[indices]
        
        return embeddings
    
    def augment_temporal_target(self, target, song_length, max_shift_frames=5):
        """
        Aplica pequeños desplazamientos temporales al target
        """
        if torch.rand(1) < 0.3:  # 30% probabilidad
            shift = torch.randint(-max_shift_frames, max_shift_frames + 1, (1,)).item()
            target_frames = target * song_length
            new_target_frames = torch.clamp(target_frames + shift, 0, song_length - 1)
            return new_target_frames / song_length
        return target

def calculate_metrics(predictions, targets, confidences, time_thresholds=[1, 3, 5, 10]):
    """
    Calcula métricas detalladas de evaluación
    predictions, targets: tensores normalizados [0,1]
    confidences: valores de confianza [0,1]
    time_thresholds: umbrales en segundos para calcular recall
    """
    # Convertir a segundos (asumiendo que targets están normalizados respecto a duración)
    # Necesitarás pasar las duraciones reales para hacer esto correctamente
    
    errors_norm = torch.abs(predictions - targets)
    
    metrics = {}
    
    # MAE en términos normalizados
    metrics['mae_normalized'] = torch.mean(errors_norm).item()
    
    # Métricas de confianza
    metrics['mean_confidence'] = torch.mean(confidences).item()
    metrics['confidence_std'] = torch.std(confidences).item()
    
    # Correlación entre confianza y precisión
    accuracy = 1.0 - errors_norm  # Mayor accuracy = menor error
    if len(confidences) > 1:
        correlation = torch.corrcoef(torch.stack([confidences, accuracy]))[0,1]
        metrics['confidence_accuracy_correlation'] = correlation.item()
    
    # Recall at different thresholds (necesita las duraciones reales)
    for threshold_seconds in time_thresholds:
        # Aproximación: asumimos canciones de ~200 segundos promedio
        threshold_norm = threshold_seconds / 200.0
        recall = torch.mean((errors_norm < threshold_norm).float()).item()
        metrics[f'recall_at_{threshold_seconds}s'] = recall
    
    # Métricas de calibración de confianza
    # Dividir en bins de confianza y ver accuracy real
    n_bins = 5
    conf_bins = torch.linspace(0, 1, n_bins + 1)
    calibration_error = 0
    
    for i in range(n_bins):
        bin_mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
        if bin_mask.sum() > 0:
            bin_confidence = confidences[bin_mask].mean()
            bin_accuracy = (errors_norm[bin_mask] < 0.05).float().mean()  # 5% como threshold
            calibration_error += torch.abs(bin_confidence - bin_accuracy) * bin_mask.sum()
    
    metrics['expected_calibration_error'] = (calibration_error / len(confidences)).item()
    
    return metrics

def adaptive_threshold_selection(confidences, errors, target_precision=0.8):
    """
    Selecciona umbral de confianza para alcanzar precisión objetivo
    """
    # Ordenar por confianza descendente
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_conf = confidences[sorted_indices]
    sorted_errors = errors[sorted_indices]
    
    # Encontrar umbral que da la precisión deseada
    for i in range(1, len(sorted_conf) + 1):
        precision = torch.mean((sorted_errors[:i] < 0.05).float())  # 5% threshold
        if precision >= target_precision:
            return sorted_conf[i-1].item()
    
    return 0.5  # Fallback

def print_training_progress(epoch, train_loss, val_metrics, best_mae):
    """Imprime progreso de entrenamiento de forma clara"""
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}")
    print(f"{'='*60}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val MAE (norm): {val_metrics['mae_normalized']:.4f}")
    print(f"Mean Confidence: {val_metrics['mean_confidence']:.3f} ± {val_metrics['confidence_std']:.3f}")
    
    if 'confidence_accuracy_correlation' in val_metrics:
        print(f"Conf-Accuracy Correlation: {val_metrics['confidence_accuracy_correlation']:.3f}")
    
    print("Recall at thresholds:")
    for key, value in val_metrics.items():
        if 'recall_at' in key:
            print(f"  {key}: {value:.2%}")
    
    print(f"Calibration Error: {val_metrics['expected_calibration_error']:.4f}")
    
    if val_metrics['mae_normalized'] < best_mae:
        print("🎉 NEW BEST MODEL!")
    
    print(f"{'='*60}\n")