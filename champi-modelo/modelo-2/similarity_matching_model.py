import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityMatchingModel(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        
        # Encoders para procesar embeddings
        self.clip_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.song_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Proyección para comparación
        self.projection = nn.Linear(embed_dim, hidden_dim)
        
        # Red para generar confianza basada en similitud
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def encode_sequences(self, clip, song):
        """Codifica clip y canción"""
        # Agregar encoding posicional si no está presente
        clip_encoded = self.clip_encoder(clip)  # [B, T_clip, 512]
        song_encoded = self.song_encoder(song)  # [B, T_song, 512]
        
        # Proyectar a espacio de comparación
        clip_proj = self.projection(clip_encoded)  # [B, T_clip, hidden_dim]
        song_proj = self.projection(song_encoded)  # [B, T_song, hidden_dim]
        
        return clip_proj, song_proj
    
    def compute_similarity_matrix(self, clip_proj, song_proj):
        """
        Computa matriz de similitud entre clip y canción
        clip_proj: [B, T_clip, H]
        song_proj: [B, T_song, H]
        Returns: [B, T_song] - similitud promedio para cada posición en la canción
        """
        batch_size, T_clip, hidden_dim = clip_proj.shape
        T_song = song_proj.shape[1]
        
        # Normalizar para similitud coseno
        clip_norm = F.normalize(clip_proj, p=2, dim=-1)  # [B, T_clip, H]
        song_norm = F.normalize(song_proj, p=2, dim=-1)  # [B, T_song, H]
        
        # Computar similitud por ventanas deslizantes
        similarities = []
        
        for i in range(T_song - T_clip + 1):
            # Ventana de la canción del mismo tamaño que el clip
            song_window = song_norm[:, i:i+T_clip, :]  # [B, T_clip, H]
            
            # Similitud elemento a elemento
            sim = torch.sum(clip_norm * song_window, dim=-1)  # [B, T_clip]
            # Promedio de similitud en la ventana
            avg_sim = torch.mean(sim, dim=1)  # [B]
            similarities.append(avg_sim)
        
        # Si la canción es más corta que el clip, rellenar con similitud baja
        if len(similarities) == 0:
            similarities = [torch.zeros(batch_size, device=clip_proj.device)]
        
        # Concatenar todas las similitudes
        similarity_scores = torch.stack(similarities, dim=1)  # [B, T_song-T_clip+1]
        
        # Pad para que tenga el mismo tamaño que T_song
        if similarity_scores.shape[1] < T_song:
            padding = torch.zeros(batch_size, T_song - similarity_scores.shape[1], 
                                device=similarity_scores.device)
            similarity_scores = torch.cat([similarity_scores, padding], dim=1)
        
        return similarity_scores
    
    def forward(self, clip, song):
        """
        clip: [B, T_clip, 512]
        song: [B, T_song, 512]
        """
        # Codificar secuencias
        clip_proj, song_proj = self.encode_sequences(clip, song)
        
        # Computar similitudes
        similarity_scores = self.compute_similarity_matrix(clip_proj, song_proj)  # [B, T_song]
        
        # Encontrar posición de máxima similitud
        max_positions = torch.argmax(similarity_scores, dim=1)  # [B]
        max_similarities = torch.max(similarity_scores, dim=1)[0]  # [B]
        
        # Normalizar posición a [0, 1]
        T_song = similarity_scores.shape[1]
        normalized_positions = max_positions.float() / max(T_song - 1, 1)  # [B]
        
        # Generar características para confianza
        batch_size = clip_proj.shape[0]
        
        # Promedio del clip y máxima similitud de la canción
        clip_avg = torch.mean(clip_proj, dim=1)  # [B, H]
        
        # Obtener representación de la posición encontrada
        song_at_max = []
        for b in range(batch_size):
            pos = max_positions[b].item()
            if pos < song_proj.shape[1]:
                song_at_max.append(song_proj[b, pos, :])
            else:
                song_at_max.append(torch.zeros_like(song_proj[b, 0, :]))
        song_at_max = torch.stack(song_at_max)  # [B, H]
        
        # Concatenar características para confianza
        conf_features = torch.cat([clip_avg, song_at_max], dim=-1)  # [B, 2*H]
        confidence = self.confidence_net(conf_features).squeeze(-1)  # [B]
        
        # Ajustar confianza basada en similitud máxima
        confidence = confidence * torch.sigmoid(max_similarities * 5)  # Amplificar señal
        
        return normalized_positions, confidence


class EnhancedSimilarityModel(nn.Module):
    """Versión mejorada con attention para mejor localización"""
    def __init__(self, embed_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        
        # Encoders más profundos
        self.clip_encoder = self._build_encoder(embed_dim, hidden_dim, num_heads, num_layers=3)
        self.song_encoder = self._build_encoder(embed_dim, hidden_dim, num_heads, num_layers=3)
        
        # Cross-attention para localización
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Localización basada en attention weights
        self.localization_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Red de confianza más sofisticada
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def _build_encoder(self, embed_dim, hidden_dim, num_heads, num_layers):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, clip, song):
        """
        clip: [B, T_clip, 512]
        song: [B, T_song, 512]
        """
        batch_size = clip.shape[0]
        
        # Codificar secuencias
        clip_encoded = self.clip_encoder(clip)  # [B, T_clip, 512]
        song_encoded = self.song_encoder(song)  # [B, T_song, 512]
        
        # Cross-attention: clip busca en canción
        attended_clip, attention_weights = self.cross_attention(
            query=clip_encoded,
            key=song_encoded,
            value=song_encoded
        )  # attended_clip: [B, T_clip, 512], attention_weights: [B, T_clip, T_song]
        
        # Promediar pesos de attention sobre T_clip para obtener distribución sobre T_song
        song_attention = torch.mean(attention_weights, dim=1)  # [B, T_song]
        
        # Encontrar posición de máxima atención
        max_positions = torch.argmax(song_attention, dim=1)  # [B]
        max_attention = torch.max(song_attention, dim=1)[0]  # [B]
        
        # Normalizar posición
        T_song = song_attention.shape[1]
        normalized_positions = max_positions.float() / max(T_song - 1, 1)
        
        # Características para confianza
        clip_repr = torch.mean(attended_clip, dim=1)  # [B, 512]
        
        # Representación de la canción en la posición encontrada
        song_repr = []
        for b in range(batch_size):
            pos = max_positions[b].item()
            if pos < song_encoded.shape[1]:
                song_repr.append(song_encoded[b, pos, :])
            else:
                song_repr.append(torch.zeros_like(song_encoded[b, 0, :]))
        song_repr = torch.stack(song_repr)  # [B, 512]
        
        # Confianza basada en representaciones y atención
        conf_input = torch.cat([clip_repr, song_repr], dim=-1)  # [B, 1024]
        confidence = self.confidence_net(conf_input).squeeze(-1)  # [B]
        
        # Ajustar confianza con intensidad de atención
        confidence = confidence * torch.sigmoid(max_attention * 3)
        
        return normalized_positions, confidence