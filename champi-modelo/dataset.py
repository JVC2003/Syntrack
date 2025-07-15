#dataset.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class AudioMatchingDataset(Dataset):
    def __init__(self, excel_path, clips_dir, songs_dir, frame_rate=10):
        self.data = pd.read_excel(excel_path)
        self.clips_dir = clips_dir
        self.songs_dir = songs_dir
        self.frame_rate = frame_rate  # OpenL3: 0.1s hop → 10 frames/segundo

    def parse_time_to_seconds(self, time_str):
        """
        Convierte tiempo tipo '1:39' (minutos:segundos) → 99.0 segundos
        """
        if isinstance(time_str, (int, float)):
            return float(time_str)
        parts = str(time_str).split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 1:
            return float(parts[0])
        else:
            raise ValueError(f"Formato de tiempo no reconocido: {time_str}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        audio_base_raw = row['audio_base']
        song_id = os.path.splitext(audio_base_raw)[0]
        folder = row['carpeta'].replace('clips_', '')
        clip_file = row['archivo_clip']
        clip_file = f"{folder}_{os.path.splitext(clip_file)[0]}.npz"
        clip_id = f"{song_id}_{folder}_{clip_file}"

        start_seconds = self.parse_time_to_seconds(row['segundo_inicio'])

        # Construcción de ID y paths
        
        

        song_path = os.path.join(
            self.songs_dir,
            f"{song_id}_openl3_512d_48000sr_0.1s.npz"
        )

        clip_path = os.path.join(
            self.clips_dir,
            song_id,
            clip_file
        )

        # Cargar embeddings desde archivos .npz
        song_embed = np.load(song_path)['emb']  # [T_song, 512]
        clip_embed = np.load(clip_path)['emb']  # [T_clip, 512]

        # Convertir a posición normalizada
        start_frame = round(start_seconds * self.frame_rate)
        target = start_frame / song_embed.shape[0]

        return {
            'clip_id': clip_id,
            'clip': torch.tensor(clip_embed, dtype=torch.float32),
            'song': torch.tensor(song_embed, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'valid': start_seconds >= 0
        }
