from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

def predict_timestamp_from_embeddings(clip_embed_path, song_embed_path):
    """
    clip_embed_path: ruta al archivo .npz del clip
    song_embed_path: ruta al archivo .npz de la canción de estudio
    """
    clip_data = np.load(clip_embed_path)
    song_data = np.load(song_embed_path)
    
    emb_clip = clip_data["emb"]
    emb_song = song_data["emb"]
    ts_song = song_data["ts"]

    # Comparación por distancia
    dist = cdist(emb_clip, emb_song, metric="euclidean")
    best_per_frame = np.argmin(dist, axis=1)
    best_match = np.argmax(np.bincount(best_per_frame))

    return float(ts_song[best_match])

def parse_time_to_seconds(time_str):
    """Convierte formato tipo '1:39' a segundos."""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = str(time_str).split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"Formato de tiempo no reconocido: {time_str}")