"""
Progressive Multi-Branch Multimodal Architecture for IV Infiltration Monitoring.

Modular Neural Sequence:
- Model 1: Baseline Causal TCN (Local Temp + Pressure time series)
- Model 2: Bioimpedance-Augmented TCN (TCN + Bioimpedance Feature Encoder)
- Model 3: Stress-Aware Multimodal Model (TCN + Bioimpedance + WESAD-derived Systemic Stress Encoder)
- Model 4: Age-Aware Multimodal Model (TCN + Bioimpedance + Stress + Modular Age Encoder)

Design Requirements:
- Strictly causal temporal encoder (no future leakage).
- Modular encoders that can be frozen, pre-trained, or ablated.
- Cross-modal fusion preserving individual stream semantics.
- Full compatibility with existing TCN baseline and WESAD encoder weights.
"""

import math
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tcn import _TCNResidualBlock


# =============================================================================
# 1. Modular Branch Encoders
# =============================================================================

class CausalTCNEncoder(nn.Module):
    """
    Causal Dilated Temporal Convolutional Network Encoder for Local Transducers.
    Takes (B, C_in, L) where C_in=2 (temperature, pressure) and L=50 (1 Hz).
    Receptive field: 61 seconds (RF > window length).
    Outputs a latent local temporal vector of dimension `latent_dim` (default 64).
    """
    def __init__(self, in_channels: int = 2, base_channels: int = 32,
                 latent_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        ch = base_channels
        self.blocks = nn.Sequential(
            _TCNResidualBlock(in_channels, ch, kernel_size=3, dilation=1, dropout=dropout),
            _TCNResidualBlock(ch, ch, kernel_size=3, dilation=2, dropout=dropout),
            _TCNResidualBlock(ch, ch * 2, kernel_size=3, dilation=4, dropout=dropout),
            _TCNResidualBlock(ch * 2, ch * 2, kernel_size=3, dilation=8, dropout=dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        if ch * 2 != latent_dim:
            self.proj = nn.Linear(ch * 2, latent_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, L)
        returns: (B, latent_dim)
        """
        h = self.blocks(x)                  # (B, 64, L)
        h = self.global_pool(h).squeeze(-1) # (B, 64)
        return self.proj(h)


class BioimpedanceEncoder(nn.Module):
    """
    Bioimpedance Feature Encoder.
    Processes multi-frequency electrical bioimpedance measurements.
    Input dimension: 28 (4 parameters [Z, R, Xc, Cm] x 7 frequencies [20..500 kHz]).
    Outputs a latent bioimpedance representation of dimension `latent_dim` (default 32).
    """
    def __init__(self, in_features: int = 28, latent_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 28)
        returns: (B, latent_dim)
        """
        return self.net(x)


class StressRepresentationEncoder(nn.Module):
    """
    Systemic Autonomic Stress Representation Encoder.
    Compatible with WESAD-trained weights.
    Processes either:
    1) Raw multi-channel wearable signals (B, 4, L_wesad): [TEMP, EDA, BVP_AMP, ACC_MAG]
    2) Pre-extracted latent stress vector (B, 32).
    Outputs a latent systemic stress vector of dimension `latent_dim` (default 32).
    """
    def __init__(self, in_channels: int = 4, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, latent_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.stress_head = nn.Linear(latent_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, L) or (B, latent_dim)
        """
        if x.dim() == 2 and x.size(1) == self.latent_dim:
            return x
        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))
        z = self.pool(h).squeeze(-1)
        return z

    def predict_stress(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward(x)
        return self.stress_head(z)


class AgeFeatureEncoder(nn.Module):
    """
    Modular Patient Age Feature Encoder.
    Models physiological variance associated with age (vascular compliance, dermal thickness).
    Takes a continuous age value (or normalized age in [0, 1]).
    Outputs a latent age embedding of dimension `latent_dim` (default 16).
    """
    def __init__(self, in_features: int = 1, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1)
        returns: (B, latent_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


# =============================================================================
# 2. Progressive Model Implementations
# =============================================================================

class Model1_BaselineTCN(nn.Module):
    """
    Model 1: Baseline Causal Temporal Convolutional Network.
    Inputs: Local Temperature time series + Local Pressure time series (B, 2, L).
    Output: 4-class logits (0: None, 1: Early, 2: Moderate, 3: Severe).
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 4,
                 base_channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.encoder = CausalTCNEncoder(in_channels=in_channels,
                                        base_channels=base_channels,
                                        latent_dim=64,
                                        dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        z_local = self.encoder(x_local) # (B, 64)
        return self.classifier(z_local)


class Model2_BioimpedanceTCN(nn.Module):
    """
    Model 2: Bioimpedance-Augmented Causal TCN.
    Inputs:
        x_local: (B, 2, L) - Temperature & Pressure
        x_bio:   (B, 28)   - Multi-frequency bioimpedance features
    Preserves the TCN temporal encoder, adds separate bioimpedance encoder,
    and fuses both latent representations before classification.
    """
    def __init__(self, in_channels: int = 2, bio_features: int = 28,
                 num_classes: int = 4, base_channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.tcn_encoder = CausalTCNEncoder(in_channels=in_channels,
                                            base_channels=base_channels,
                                            latent_dim=64,
                                            dropout=dropout)
        self.bio_encoder = BioimpedanceEncoder(in_features=bio_features,
                                               latent_dim=32,
                                               dropout=dropout)
        # Fusion layer: 64 (temporal) + 32 (bioimpedance) = 96
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(64, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_local: torch.Tensor, x_bio: torch.Tensor) -> torch.Tensor:
        z_local = self.tcn_encoder(x_local) # (B, 64)
        z_bio = self.bio_encoder(x_bio)     # (B, 32)
        fused = self.fusion(torch.cat([z_local, z_bio], dim=-1)) # (B, 64)
        return self.classifier(fused)


class Model3_StressMultimodal(nn.Module):
    """
    Model 3: Stress-Aware Multimodal Model.
    Inputs:
        x_local:  (B, 2, L) - Temperature & Pressure time series
        x_bio:    (B, 28)   - Multi-frequency bioimpedance features
        x_stress: (B, 4, L) or (B, 32) - WESAD-derived systemic autonomic signals
    Integrates:
        1) Causal TCN local encoder (64-dim)
        2) Bioimpedance encoder (32-dim)
        3) Systemic autonomic stress encoder (32-dim)
        4) Differential cross-stream thermal gating to decouple systemic cooling
           from true extravasation.
    """
    def __init__(self, in_channels: int = 2, bio_features: int = 28,
                 num_classes: int = 4, base_channels: int = 32,
                 dropout: float = 0.15, systemic_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.tcn_encoder = CausalTCNEncoder(in_channels=in_channels,
                                            base_channels=base_channels,
                                            latent_dim=64,
                                            dropout=dropout)
        self.bio_encoder = BioimpedanceEncoder(in_features=bio_features,
                                               latent_dim=32,
                                               dropout=dropout)

        if systemic_encoder is not None:
            self.stress_encoder = systemic_encoder
        else:
            self.stress_encoder = StressRepresentationEncoder(in_channels=4, latent_dim=32)

        # Cross-stream projection: Systemic stress -> expected local thermal perturbation
        self.proj_sys_to_local = nn.Linear(32, 64)

        # Differential gating: how much of the local signal is uncoupled local pathology
        self.gate_net = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )

        # Multimodal fusion: Gated Local (64) + Bioimpedance (32) + Systemic Stress (32) = 128
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(64, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_local: torch.Tensor, x_bio: torch.Tensor,
                x_stress: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_local = self.tcn_encoder(x_local) # (B, 64)
        z_bio = self.bio_encoder(x_bio)     # (B, 32)

        if x_stress is not None:
            z_stress = self.stress_encoder(x_stress) # (B, 32)
        else:
            z_stress = torch.zeros(z_local.size(0), 32, device=z_local.device, dtype=z_local.dtype)

        # Decouple systemic vasoconstriction
        expected_sys_effect = self.proj_sys_to_local(z_stress)
        residual_local = z_local - expected_sys_effect
        gate = self.gate_net(torch.cat([z_local, z_stress], dim=-1))
        gated_local = gate * residual_local

        # Full tripartite multimodal fusion
        fused = self.fusion(torch.cat([gated_local, z_bio, z_stress], dim=-1))
        return self.classifier(fused)


class Model4_AgeAwareMultimodal(nn.Module):
    """
    Model 4: Age-Aware Multimodal Neural Architecture.
    Inputs:
        x_local:  (B, 2, L) - Temperature & Pressure
        x_bio:    (B, 28)   - Multi-frequency bioimpedance
        x_stress: (B, 4, L) or (B, 32) - Systemic stress
        x_age:    (B, 1)    - Normalized patient age
    Integrates all previous encoders plus the modular AgeFeatureEncoder.
    Designed for verification of modular architectural extensibility.
    """
    def __init__(self, in_channels: int = 2, bio_features: int = 28,
                 num_classes: int = 4, base_channels: int = 32,
                 dropout: float = 0.15, systemic_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.tcn_encoder = CausalTCNEncoder(in_channels=in_channels,
                                            base_channels=base_channels,
                                            latent_dim=64,
                                            dropout=dropout)
        self.bio_encoder = BioimpedanceEncoder(in_features=bio_features,
                                               latent_dim=32,
                                               dropout=dropout)

        if systemic_encoder is not None:
            self.stress_encoder = systemic_encoder
        else:
            self.stress_encoder = StressRepresentationEncoder(in_channels=4, latent_dim=32)

        self.age_encoder = AgeFeatureEncoder(in_features=1, latent_dim=16)

        self.proj_sys_to_local = nn.Linear(32, 64)
        self.gate_net = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )

        # Fusion: Gated Local (64) + Bioimpedance (32) + Stress (32) + Age (16) = 144
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32 + 16, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(64, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_local: torch.Tensor, x_bio: torch.Tensor,
                x_stress: Optional[torch.Tensor] = None,
                x_age: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_local = self.tcn_encoder(x_local)
        z_bio = self.bio_encoder(x_bio)

        if x_stress is not None:
            z_stress = self.stress_encoder(x_stress)
        else:
            z_stress = torch.zeros(z_local.size(0), 32, device=z_local.device, dtype=z_local.dtype)

        if x_age is not None:
            z_age = self.age_encoder(x_age)
        else:
            z_age = torch.zeros(z_local.size(0), 16, device=z_local.device, dtype=z_local.dtype)

        expected_sys_effect = self.proj_sys_to_local(z_stress)
        residual_local = z_local - expected_sys_effect
        gate = self.gate_net(torch.cat([z_local, z_stress], dim=-1))
        gated_local = gate * residual_local

        fused = self.fusion(torch.cat([gated_local, z_bio, z_stress, z_age], dim=-1))
        return self.classifier(fused)
