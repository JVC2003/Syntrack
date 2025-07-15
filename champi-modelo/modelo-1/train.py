# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict
import random

from dataset import AudioMatchingDataset
from cross_attencion import CrossAttentionRegressor
from utils import collate_fn, loss_fn

# Config
BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
dataset = AudioMatchingDataset(
    excel_path='clips_syntrack/super_etiquetado_Wowow.xlsx',
    clips_dir='embeddings/clips',
    songs_dir='embeddings/songs'
)

# Split por canción
by_song = defaultdict(list)
for idx, sample in enumerate(dataset):
    song_key = sample['clip_id'].split("_clip")[0]
    by_song[song_key].append(idx)

all_keys = list(by_song.keys())
random.shuffle(all_keys)

val_keys = set(all_keys[:int(0.2 * len(all_keys))])
train_keys = set(all_keys[int(0.2 * len(all_keys)):])

train_indices = [i for k in train_keys for i in by_song[k]]
val_indices = [i for k in val_keys for i in by_song[k]]

train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Modelo
model = CrossAttentionRegressor().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
seconds_margin = 5
frame_margin = seconds_margin * 10  # asumiendo 10 fps

# Entrenamiento
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        clip = batch['clip'][0].unsqueeze(0).to(DEVICE)  
        song = batch['song'][0].unsqueeze(0).to(DEVICE)  
        target = batch['target'].to(DEVICE)  
        valid = batch['valid']

        if not valid:
            continue

        ts_pred, conf_pred = model(clip, song)
        loss, loss_dict = loss_fn(ts_pred, conf_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {running_loss / len(train_loader):.4f}")

    # Validación
    model.eval()
    mae_total = 0
    recall_at_1s = 0
    with torch.no_grad():
        for batch in val_loader:
            clip = batch['clip'][0].unsqueeze(0).to(DEVICE)
            song = batch['song'][0].unsqueeze(0).to(DEVICE)
            target = batch['target'].to(DEVICE)
            valid = batch['valid']

            if not valid:
                continue

            ts_pred, conf_pred = model(clip, song)
            error = torch.abs(ts_pred - target)

            mae_total += error.item()
            recall_at_1s += (error < frame_margin / song.shape[1]).item()

    val_mae = mae_total / len(val_loader)
    val_recall = recall_at_1s / len(val_loader)

    print(f"[Epoch {epoch+1}] Val MAE: {val_mae:.4f} | Recall@1s: {val_recall:.2%}")

torch.save(model.state_dict(), "best_model.pth")
print("Modelo guardado en best_model.pth")
