import torch
import torch.nn as nn

class CrossAttentionRegressor(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256):
        super().__init__()

        self.clip_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.song_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.attn = nn.MultiheadAttention(embed_dim=2*hidden_dim, num_heads=4, batch_first=True)

        # Pooling + regresión
        self.pool = lambda x: torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)  # [B, 4H]

        self.regressor = nn.Sequential(
            nn.Linear(4 * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # [timestamp, confidence]
        )

    def forward(self, clip, song):
        """
        clip: (B, T1, 512)
        song: (B, T2, 512)
        """
        # Encoder
        clip_encoded, _ = self.clip_encoder(clip)  # [B, T1, 2H]
        song_encoded, _ = self.song_encoder(song)  # [B, T2, 2H]

        # Cross-attention
        context, _ = self.attn(query=clip_encoded, key=song_encoded, value=song_encoded)  # [B, T1, 2H]

        # Pooling
        pooled = self.pool(context)  # [B, 4H]

        # Regressor
        out = self.regressor(pooled)  # [B, 2]
        timestamp = torch.sigmoid(out[:, 0])      # ∈ [0, 1]
        confidence = torch.sigmoid(out[:, 1])     # ∈ [0, 1]

        return timestamp, confidence
