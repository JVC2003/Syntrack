#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sincroniza un clip con la versión de estudio usando OpenL3.
Incluye caché opcional del embedding de la canción de estudio.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import openl3
import librosa
from scipy.spatial.distance import cdist

# -------------------- CLI --------------------
parser = argparse.ArgumentParser(
    description="Sincroniza un clip con la versión de estudio usando embeddings OpenL3."
)
parser.add_argument("--song-id", default="clocks_coldplay",
                    help="Carpeta/prefijo de la canción (default: clocks_coldplay)")
parser.add_argument("--clip-num", type=int, default=4,
                    help="Número de clip, ej. 4 para *_clip04* (default: 4)")
parser.add_argument("--no-extract", action="store_true",
                    help="Omitir la extracción de audio incluso si el WAV no existe")
parser.add_argument("--force-extract", action="store_true",
                    help="Extraer siempre, incluso si el WAV ya está creado")
parser.add_argument("--cache-studio-embed", action="store_true",
                    help="Guardar/usar un .npz con el embedding de la canción de estudio")
parser.add_argument("--refresh-studio-embed", action="store_true",
                    help="Recalcular y sobrescribir el embedding guardado (requiere --cache-studio-embed)")
args = parser.parse_args()

# -------------------- Parámetros fijos --------------------
SR_TARGET = 48_000
HOP_SIZE  = 0.1
EMB_SIZE  = 512

# -------------------- Rutas --------------------
base_dir   = Path("clips_syntrack") / args.song_id
studio_mp3 = base_dir / "cancion_estudio" / f"{args.song_id}.mp3"
video_mp4  = base_dir / "clips_youtube"   / f"{args.song_id}_clip{args.clip_num:02d}.mp4"
wav_dir    = base_dir / "clips_youtube"   / "wavs"
clip_wav   = wav_dir / f"{args.song_id}_clip{args.clip_num:02d}.wav"

# Archivo de caché: mismo nombre de la canción + parámetros clave
embed_cache = studio_mp3.with_suffix("").with_name(
    f"{args.song_id}_openl3_{EMB_SIZE}d_{SR_TARGET}sr_{HOP_SIZE}s.npz"
)

# -------------------- Comprobaciones --------------------
if not studio_mp3.exists():
    sys.exit(f"❌ Canción de estudio no encontrada:\n   {studio_mp3.resolve()}")

if not clip_wav.exists() and not args.force_extract and args.no_extract:
    sys.exit("❌ El WAV no existe y se pidió --no-extract. Nada que hacer.")

# -------------------- Extracción opcional --------------------
if args.force_extract or (not clip_wav.exists() and not args.no_extract):
    print("1️⃣  Extrayendo audio del video...")
    try:
        import moviepy as mp
    except ModuleNotFoundError:
        sys.exit("⚠️  MoviePy no está instalado (`pip install moviepy`) y no hay WAV. Aborto.")
    wav_dir.mkdir(parents=True, exist_ok=True)
    mp.VideoFileClip(str(video_mp4)).audio.write_audiofile(str(clip_wav))
else:
    print("1️⃣  Extracción omitida.")

# -------------------- Carga de audio --------------------
print("2️⃣  Cargando audios...")
song, sr_song = librosa.load(studio_mp3, sr=SR_TARGET)
clip, sr_clip = librosa.load(clip_wav,  sr=SR_TARGET)
clip = librosa.util.normalize(clip)

# -------------------- Embedding de la canción (con caché) --------------------
if args.cache_studio_embed and embed_cache.exists() and not args.refresh_studio_embed:
    print("3️⃣  Cargando embedding de la canción desde caché...")
    data = np.load(embed_cache)
    emb_song = data["emb"]
    ts_song  = data["ts"]
else:
    print("3️⃣  Extrayendo embedding de la canción con OpenL3...")
    emb_song, ts_song = openl3.get_audio_embedding(
        song, sr_song,
        input_repr="mel256", content_type="music",
        embedding_size=EMB_SIZE, hop_size=HOP_SIZE
    )
    if args.cache_studio_embed:
        np.savez_compressed(embed_cache, emb=emb_song, ts=ts_song)
        print(f"Embedding guardado en {base_dir / 'cancion_estudio' / embed_cache.name}")

# -------------------- Embedding del clip (siempre) --------------------
print("4️⃣  Extrayendo embedding del clip con OpenL3...")
emb_clip, ts_clip = openl3.get_audio_embedding(
    clip, sr_clip,
    input_repr="mel256", content_type="music",
    embedding_size=EMB_SIZE, hop_size=HOP_SIZE
)

# -------------------- Comparación --------------------
print("5️⃣  Comparando embeddings...")
dist = cdist(emb_clip, emb_song, metric="euclidean")
best_per_frame = np.argmin(dist, axis=1)
best_match = np.argmax(np.bincount(best_per_frame))

print(f"\n🎯  El clip coincide mejor con el segundo {int(ts_song[best_match])} de la canción original.")
