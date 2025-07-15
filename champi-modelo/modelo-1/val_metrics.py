# val_metrics.py
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import random

from dataset import AudioMatchingDataset
from cross_attencion import CrossAttentionRegressor
from utils import collate_fn

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
seconds_margin = 6  # para recall@6s
frame_margin = seconds_margin * 10

# Dataset
dataset = AudioMatchingDataset(
    excel_path='clips_syntrack/super_etiquetado_Wowow.xlsx',
    clips_dir='embeddings/clips',
    songs_dir='embeddings/songs'
)

# Split por canción (mismo método que en train.py)
by_song = defaultdict(list)
for idx, sample in enumerate(dataset):
    song_key = sample['clip_id'].split("_clip")[0]
    by_song[song_key].append(idx)

all_keys = list(by_song.keys())
random.seed(42)  # para reproducibilidad
random.shuffle(all_keys)

val_keys = set(all_keys[:int(0.2 * len(all_keys))])
val_indices = [i for k in val_keys for i in by_song[k]]
val_set = Subset(dataset, val_indices)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Modelo
model = CrossAttentionRegressor().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# Métricas
mae_total = 0
recall_margin_count = 0
y_true = []
y_pred = []

total_valid = 0

with torch.no_grad():
    for batch in val_loader:
        clip_id = batch['clip_id'][0]
        clip = batch['clip'][0].unsqueeze(0).to(DEVICE)
        song = batch['song'][0].unsqueeze(0).to(DEVICE)
        target = batch['target'].to(DEVICE)
        valid = batch['valid']

        if not valid:
            continue

        ts_pred, conf_pred = model(clip, song)
        error = torch.abs(ts_pred - target)

        mae_total += error.item()
        recall_margin_count += (error < frame_margin / song.shape[1]).item()

        y_true.append(int(target.item() * 10))  # discretizado a frames
        y_pred.append(int(ts_pred.item() * 10))

        print(f"[{clip_id}] Target: {target.item() * song.shape[1]:.1f}s | Pred: {ts_pred.item() * song.shape[1]:.1f}s | Conf: {conf_pred.item():.2f} | Error: {error.item() * song.shape[1]:.1f}s")
        total_valid += 1

# Reporte
if total_valid > 0:
    val_mae = mae_total / total_valid
    recall_at_margin = recall_margin_count / total_valid
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    print(f"[Val] Total: {len(val_loader)} | Válidos: {total_valid}")
    print(f"MAE: {val_mae * song.shape[1]:.2f} s")
    print(f"Recall@{seconds_margin}s: {recall_at_margin:.2%}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
else:
    print("No hay ejemplos válidos en validación.")
