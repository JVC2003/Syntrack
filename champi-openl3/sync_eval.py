import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.spatial.distance import cdist
from utils import *

# Rutas base de los embeddings
EXCEL_PATH = Path("data.xlsx")
CLIPS_DIR = Path("embeddings/clips_renombrados")
SONGS_DIR = Path("embeddings/songs")
EMBED_DIM = 512
SR = 48000
HOP = 0.1

# Cargar CSV
df = pd.read_excel("data.xlsx")

# Para guardar los resultados
predictions = []

for idx, row in df.iterrows():
    try:
        # Rutas
        audio_base_raw = row['audio_base']
        song_id = os.path.splitext(audio_base_raw)[0]
        folder = row['carpeta']
        clip_file = row['archivo_clip']
        clip_id = f"{song_id}_{folder}_{clip_file}"

        start_seconds = parse_time_to_seconds(row['segundo_inicio'])
        
        clip_path = CLIPS_DIR / song_id / f"{folder}_{clip_file}.npz"
        song_path = SONGS_DIR / f"{song_id}_openl3_512d_48000sr_0.1s.npz"
        
        # Cargar embeddings
        timestamp = predict_timestamp_from_embeddings(clip_path,song_path)
        print(f"🎯 Clip: {clip_id} | ✅ Etiqueta: {start_seconds:.2f}s | 🤖 Predicción: {timestamp:.2f}s")
        predictions.append(timestamp)
    except Exception as e:
        print(f"⚠️ Error en fila {idx} ({clip_file}): {e}")
        predictions.append("error")

# Guardar resultados
df["segundo_predecido"] = predictions
df['segundo_inicio'] = df['segundo_inicio'].apply(parse_time_to_seconds)
df.to_excel("results2.xlsx", index=False)
print("✅ Resultados guardados en 'results.xlsx")

