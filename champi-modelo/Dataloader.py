from dataset import *
from cross_attencion import *
import os

dataset = AudioMatchingDataset(
    excel_path='clips_syntrack\super_etiquetado_Wowow.xlsx',
    clips_dir='embeddings/clips',
    songs_dir='embeddings/songs'
)

#sample = dataset[0]
for sample in dataset:
    print("Clip ID:", sample['clip_id'])
    print("Clip shape:", sample['clip'].shape)
    print("Song shape:", sample['song'].shape)
    print("Target timestamp (normalized):", sample['target'].item())

model = CrossAttentionRegressor()
clip = torch.randn(1, 30, 512)   # 3 segundos
song = torch.randn(1, 600, 512)  # 1 minuto

ts_pred, conf_pred = model(clip, song)
print("Timestamp predicho:", ts_pred.item())
print("Confianza:", conf_pred.item())