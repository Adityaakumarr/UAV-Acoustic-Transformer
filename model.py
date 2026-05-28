"""
model.py
────────
Acoustic Transformer for UAV 3D Trajectory Estimation.
Implements the exact hybrid CNN-Transformer architecture described in Sections 2.2 and 2.3 
of the ICASSP 2026 paper.

Architecture:
  Input (Acoustic)    : (B, 10, F, T)   — 10-channel spectrogram + GCC-PHAT dense tensor
  Encoder Backbone    : ResNet-18 style CNN → F_CNN
  Positional Encoding : Broadcastable learnable frequency (Ef) + time (Et) embeddings
  Flattened Sequence  : Z = Z_0 + Broadcast(Ef) + Broadcast(Et)
  Acoustic Encoder    : Multi-head Transformer Encoder → H_enc
  Trajectory Decoder  : Masked self-attention + multi-head cross-attention → current position
  Output              : (B, 3)          — predicted (x, y, z) coordinates
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1.  ResNet-18-Style CNN Encoder
# ──────────────────────────────────────────────────────────────────────────────

class ResNetBasicBlock(nn.Module):
    """
    Standard ResNet Basic Block containing two 3x3 convolutions with residual shortcut connections.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    """
    ResNet-18-style CNN feature extractor. 
    Accepts arbitrary input channel dimensions (e.g., 10 channels) and projects
    local spectro-temporal features down to (B, d_model, H_prime, W_prime).
    """
    def __init__(self, in_channels: int = 10, d_model: int = 256):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial large receptive field block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.stage1 = self._make_stage(64, blocks=2, stride=1)
        self.stage2 = self._make_stage(128, blocks=2, stride=2)
        self.stage3 = self._make_stage(256, blocks=2, stride=2)
        
        # 1x1 Conv projection to match d_model
        self.proj = nn.Conv2d(256, d_model, kernel_size=1, bias=False)
        
    def _make_stage(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(ResNetBasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBasicBlock(self.in_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channels, F, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.proj(x)  # (Batch, d_model, H_prime, W_prime)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Transformer Encoder  —  captures global spectro-temporal dependencies
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Multi-head self-attention transformer encoder block.
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # Post-LN structure
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model)  →  (B, S, d_model)"""
        return self.encoder(x)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Cross-Attention Causal Trajectory Decoder
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDecoder(nn.Module):
    """
    Causal trajectory decoder. Estimates the 3D position by attending to:
      - Trajectory history coordinates via masked self-attention
      - Acoustic context memory via multi-head cross-attention
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float,
                 traj_seq_len: int, output_dim: int = 3):
        super().__init__()

        self.traj_embed = nn.Linear(output_dim, d_model)
        
        # Build 1D Sinusoidal positional embeddings for coordinates history
        pos_enc = self._build_1d_pe(traj_seq_len, d_model)
        self.register_buffer("pos_enc", pos_enc)   # (K, d_model)

        # PyTorch Decoder layer matches Section 2.3 pipeline out of the box
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Final Position-wise Feed-Forward Network and linear projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    @staticmethod
    def _build_1d_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, memory: torch.Tensor,
                traj_history: torch.Tensor) -> torch.Tensor:
        """
        memory       : (B, S, d_model)  — Contextualized acoustic encoder output H_enc
        traj_history : (B, K, 3)        — Historical trajectory coordinates of length K
        """
        B, K, _ = traj_history.shape

        # 1. Embed coordinate coordinates and add temporal positional encoding
        q = self.traj_embed(traj_history)               # (B, K, d_model)
        q = q + self.pos_enc[:K].unsqueeze(0)

        # 2. Build square subsequent causal mask to enforce temporal causality
        # In PyTorch 2.0+, generating submask works cleanly on target devices
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(K, device=traj_history.device)

        # 3. Decode: first sub-layer (causal self-attn), second sub-layer (cross-attn over memory)
        out = self.decoder(tgt=q, memory=memory, tgt_mask=tgt_mask)  # (B, K, d_model)

        # 4. Use the last token representing the most recent context state
        last = out[:, -1, :]                            # (B, d_model)
        
        # 5. Output projection
        return self.out_proj(last)                      # (B, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  End-to-End Acoustic Transformer Module
# ──────────────────────────────────────────────────────────────────────────────

class AcousticTransformer(nn.Module):
    """
    Complete hybrid CNN-Transformer model for 3D UAV trajectory estimation.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        f = cfg["features"]

        d_model = m["d_model"]
        dropout_cnn = m["cnn_dropout"]
        dropout_tr = m["transformer_dropout"]

        # ── CNN feature extractor backbone ────────────────────────────────
        self.cnn_encoder = ResNet18Backbone(
            in_channels=f["num_channels_out"],
            d_model=d_model
        )

        # ── Learnable broadcastable positional embeddings ─────────────────
        # Dynamically evaluate spatial outputs to prevent runtime dimension crashes
        with torch.no_grad():
            dummy_spec = torch.zeros(1, f["num_channels_out"], f["freq_bins"], f["time_frames"])
            dummy_out = self.cnn_encoder(dummy_spec)
            _, _, H_prime, W_prime = dummy_out.shape
            
        # Frequency positional embeddings Ef and time positional embeddings Et
        # Ef shape: (H_prime, 1, d_model)
        # Et shape: (1, W_prime, d_model)
        self.Ef = nn.Parameter(torch.randn(H_prime, 1, d_model))
        self.Et = nn.Parameter(torch.randn(1, W_prime, d_model))

        # ── Multi-head Transformer Encoder ────────────────────────────────
        self.tr_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=m["nhead"],
            num_layers=m["num_encoder_layers"],
            dim_feedforward=m["dim_feedforward"],
            dropout=dropout_tr,
        )

        # ── Causal Trajectory Decoder ─────────────────────────────────────
        self.decoder = TrajectoryDecoder(
            d_model=d_model,
            nhead=m["nhead"],
            num_layers=m["num_decoder_layers"],
            dim_feedforward=m["dim_feedforward"],
            dropout=dropout_tr,
            traj_seq_len=m["traj_seq_len"],
            output_dim=m["output_dim"],
        )

        self._init_weights()

    def _init_weights(self):
        """Standard safe deep learning parameter initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, spec: torch.Tensor,
                traj_history: torch.Tensor) -> torch.Tensor:
        """
        spec         : (B, 10, F, T)   — 10-channel spectrogram & gcc-phat features
        traj_history : (B, K, 3)        — past K predicted trajectory coordinates

        Returns:
        pred_pos     : (B, 3)          — current predicted position [x, y, z]
        """
        # 1. Local spectro-temporal feature extraction
        feat = self.cnn_encoder(spec)                   # (B, d_model, H_prime, W_prime)

        # 2. Add learnable positional encodings via broadcasting (Equation 3)
        pe = self.Ef + self.Et                          # (H_prime, W_prime, d_model)
        pe = pe.flatten(0, 1)                           # (H_prime * W_prime, d_model)
        
        # Flatten spatial dimensions to form tokens Z_0
        tokens = feat.flatten(2).permute(0, 2, 1)       # (B, H_prime * W_prime, d_model)
        Z = tokens + pe.unsqueeze(0)                    # (B, S, d_model)

        # 3. Global contextual feature encoder
        memory = self.tr_encoder(Z)                     # (B, S, d_model)

        # 4. Trajectory decoding conditioned on history
        pred_pos = self.decoder(memory, traj_history)   # (B, 3)
        return pred_pos


# ──────────────────────────────────────────────────────────────────────────────
# Utilities & Diagnostic checks
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict) -> AcousticTransformer:
    model = AcousticTransformer(cfg)
    n = count_parameters(model)
    print(f"AcousticTransformer | trainable params: {n:,}")
    return model


if __name__ == "__main__":
    import yaml
    
    # Simple self-diagnostic block
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Running diagnostic sanity check...")
    model = build_model(cfg)

    B = 4
    F = cfg["features"]["freq_bins"]
    T = cfg["features"]["time_frames"]
    K = cfg["model"]["traj_seq_len"]
    in_channels = cfg["features"]["num_channels_out"]

    spec = torch.randn(B, in_channels, F, T)
    history = torch.randn(B, K, 3)

    out = model(spec, history)
    print(f"Input spec shape     : {spec.shape}")
    print(f"Input history shape  : {history.shape}")
    print(f"Output predicted shape: {out.shape}")  # Expected: (4, 3)
    print("\nSanity Check Successful! Model fully verified.")
