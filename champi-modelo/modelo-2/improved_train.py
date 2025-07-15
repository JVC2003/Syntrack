import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict
import random
import numpy as np
import json
import os

from dataset import AudioMatchingDataset
from similarity_matching_model import SimilarityMatchingModel, EnhancedSimilarityModel
from improved_utils import (
    collate_fn, improved_loss_fn, focal_loss, 
    TemporalAugmenter, calculate_metrics, print_training_progress
)

# Configuración
CONFIG = {
    'batch_size': 1,  # Mantener en 1 por ahora debido a longitudes variables
    'epochs': 10,
    'lr': 5e-4,
    'weight_decay': 1e-5,
    'patience': 10,  # Early stopping
    'delta_threshold': 0.03,  # 3% de la canción como threshold de éxito
    'augmentation': True,
    'model_type': 'enhanced',  # 'basic' o 'enhanced'
    'loss_type': 'improved',  # 'improved' o 'focal'
    'save_dir': 'models',
    'log_file': 'training_log.json'
}

# Configurar dispositivo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {DEVICE}")

# Crear directorio para modelos
os.makedirs(CONFIG['save_dir'], exist_ok=True)

# Semillas para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def setup_data_loaders():
    """Configura los data loaders con split por canción"""
    dataset = AudioMatchingDataset(
        excel_path='clips_syntrack/super_etiquetado_Wowow.xlsx',
        clips_dir='embeddings/clips',
        songs_dir='embeddings/songs'
    )
    
    print(f"Dataset total: {len(dataset)} ejemplos")
    
    # Split por canción para evitar data leakage
    by_song = defaultdict(list)
    for idx, sample in enumerate(dataset):
        song_key = sample['clip_id'].split("_clip")[0]
        by_song[song_key].append(idx)
    
    all_keys = list(by_song.keys())
    random.shuffle(all_keys)
    
    # 70% train, 15% val, 15% test
    n_songs = len(all_keys)
    train_songs = all_keys[:int(0.7 * n_songs)]
    val_songs = all_keys[int(0.7 * n_songs):int(0.85 * n_songs)]
    test_songs = all_keys[int(0.85 * n_songs):]
    
    train_indices = [i for song in train_songs for i in by_song[song]]
    val_indices = [i for song in val_songs for i in by_song[song]]
    test_indices = [i for song in test_songs for i in by_song[song]]
    
    print(f"Train: {len(train_indices)} clips de {len(train_songs)} canciones")
    print(f"Val: {len(val_indices)} clips de {len(val_songs)} canciones")
    print(f"Test: {len(test_indices)} clips de {len(test_songs)} canciones")
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=CONFIG['batch_size'], 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    return train_loader, val_loader, test_set

def create_model():
    """Crea el modelo según configuración"""
    if CONFIG['model_type'] == 'enhanced':
        model = EnhancedSimilarityModel(
            embed_dim=512,
            hidden_dim=256,
            num_heads=8
        )
    else:
        model = SimilarityMatchingModel(
            embed_dim=512,
            hidden_dim=256,
            num_heads=8
        )
    
    return model.to(DEVICE)

def get_loss_function():
    """Retorna la función de pérdida según configuración"""
    if CONFIG['loss_type'] == 'improved':
        return lambda ts_pred, conf_pred, ts_true: improved_loss_fn(
            ts_pred, conf_pred, ts_true, 
            delta=CONFIG['delta_threshold'],
            conf_weight=0.3
        )
    else:
        return lambda ts_pred, conf_pred, ts_true: focal_loss(
            ts_pred, conf_pred, ts_true,
            delta=CONFIG['delta_threshold']
        )

def validate_model(model, val_loader, loss_fn, augmenter=None):
    """Evaluación completa del modelo"""
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_confidences = []
    all_errors = []
    valid_count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validando"):
            if not batch['valid'][0]:
                continue
                
            clip = batch['clip'][0].unsqueeze(0).to(DEVICE)
            song = batch['song'][0].unsqueeze(0).to(DEVICE)
            target = batch['target'].to(DEVICE)
            
            # Predicción
            ts_pred, conf_pred = model(clip, song)
            
            # Pérdida
            loss, _ = loss_fn(ts_pred, conf_pred, target)
            total_loss += loss.item()
            
            # Recopilar métricas
            all_predictions.append(ts_pred.cpu())
            all_targets.append(target.cpu())
            all_confidences.append(conf_pred.cpu())
            
            error = torch.abs(ts_pred - target).cpu()
            all_errors.append(error)
            
            valid_count += 1
    
    if valid_count == 0:
        return {'mae_normalized': float('inf')}, float('inf')
    
    # Convertir a tensores
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    confidences = torch.cat(all_confidences)
    errors = torch.cat(all_errors)
    
    # Calcular métricas
    metrics = calculate_metrics(predictions, targets, confidences)
    avg_loss = total_loss / valid_count
    
    return metrics, avg_loss

def train_epoch(model, train_loader, optimizer, loss_fn, augmenter=None):
    """Entrenamiento de una época"""
    model.train()
    total_loss = 0
    loss_components = defaultdict(list)
    valid_batches = 0
    
    for batch in tqdm(train_loader, desc="Entrenando"):
        if not batch['valid'][0]:
            continue
            
        clip = batch['clip'][0].unsqueeze(0)
        song = batch['song'][0].unsqueeze(0)
        target = batch['target']
        
        # Aplicar augmentación si está habilitada
        if augmenter and CONFIG['augmentation']:
            clip_aug = augmenter.augment_embeddings(clip.squeeze(0))
            clip = clip_aug.unsqueeze(0)
            
            # Pequeña augmentación temporal al target
            target_aug = augmenter.augment_temporal_target(
                target, song.shape[1], max_shift_frames=3
            )
            target = target_aug
        
        clip = clip.to(DEVICE)
        song = song.to(DEVICE)
        target = target.to(DEVICE)
        
        # Forward pass
        ts_pred, conf_pred = model(clip, song)
        
        # Calcular pérdida
        loss, loss_dict = loss_fn(ts_pred, conf_pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Registrar métricas
        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_components[key].append(value)
        
        valid_batches += 1
    
    # Promediar componentes de pérdida
    avg_loss = total_loss / max(valid_batches, 1)
    avg_components = {k: np.mean(v) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def main():
    """Función principal de entrenamiento"""
    print("Configurando datos...")
    train_loader, val_loader, test_set = setup_data_loaders()
    
    print("Creando modelo...")
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Optimizador y scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Función de pérdida
    loss_fn = get_loss_function()
    
    # Augmentador (si está habilitado)
    augmenter = TemporalAugmenter() if CONFIG['augmentation'] else None
    
    # Variables para early stopping y logging
    best_mae = float('inf')
    patience_counter = 0
    training_log = []
    
    print(f"\nIniciando entrenamiento por {CONFIG['epochs']} épocas...")
    print(f"Modelo: {CONFIG['model_type']}, Pérdida: {CONFIG['loss_type']}")
    print(f"Augmentación: {'Sí' if CONFIG['augmentation'] else 'No'}")
    
    for epoch in range(CONFIG['epochs']):
        # Entrenamiento
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, loss_fn, augmenter
        )
        
        # Validación
        val_metrics, val_loss = validate_model(model, val_loader, loss_fn)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['mae_normalized'])
        
        # Logging
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_components': train_components,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_log.append(epoch_log)
        
        # Imprimir progreso
        print_training_progress(epoch + 1, train_loss, val_metrics, best_mae)
        
        # Guardar mejor modelo
        if val_metrics['mae_normalized'] < best_mae:
            best_mae = val_metrics['mae_normalized']
            patience_counter = 0
            
            # Guardar modelo
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae,
                'config': CONFIG,
                'val_metrics': val_metrics
            }, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            
            print(f"Modelo guardado con MAE: {best_mae:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\nEarly stopping después de {CONFIG['patience']} épocas sin mejora")
            break
        
        # Guardar log cada 5 épocas
        if (epoch + 1) % 5 == 0:
            with open(CONFIG['log_file'], 'w') as f:
                json.dump(training_log, f, indent=2)
    
    # Guardar log final
    with open(CONFIG['log_file'], 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nEntrenamiento completado!")
    print(f"Mejor MAE: {best_mae:.4f}")
    print(f"Modelo guardado en: {CONFIG['save_dir']}/best_model.pth")
    print(f"Log guardado en: {CONFIG['log_file']}")

if __name__ == "__main__":
    main()