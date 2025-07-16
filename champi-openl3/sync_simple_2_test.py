from utils import *

timestamp = predict_timestamp_from_embeddings(
    "embeddings/clips/bzrp_session_milo_j_bizarrap/david_clip02.npz",
    "embeddings/songs/bzrp_session_milo_j_bizarrap_openl3_512d_48000sr_0.1s.npz"
)
print(f"El clip calza mejor en el segundo: {timestamp:.2f}")
