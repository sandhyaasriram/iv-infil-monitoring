"""
Autonomic-Decoupled Dual-Stream Architecture
Connects physiological representations learned from real wearable data (WESAD)
with local IV monitoring to decouple systemic vasomotor confounders from true extravasation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SystemicAutonomicEncoder(nn.Module):
    """
    Temporal encoder trained on real human multi-modal signals (WESAD wrist bundle):
    Channels: [TEMP, EDA, BVP_AMP, ACC_MAG] (4 channels).
    Extracts a 32-dimensional latent vector representing autonomic/vasomotor state.
    """
    def __init__(self, in_channels=4, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, latent_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Auxiliary heads for supervised pre-training on WESAD
        self.stress_head = nn.Linear(latent_dim, 2) # Binary: stress vs non-stress
        self.vaso_head = nn.Linear(latent_dim, 1)   # Estimated peripheral cooling

    def forward(self, x):
        # x: (B, 4, L)
        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))
        z_sys = self.pool(h).squeeze(-1) # (B, latent_dim)
        return z_sys

    def predict_stress(self, x):
        z_sys = self.forward(x)
        return self.stress_head(z_sys)


class LocalIVEstimator(nn.Module):
    """
    Temporal encoder for local IV-site transducers:
    Channels: [local_temperature, local_pressure] (2 channels).
    """
    def __init__(self, in_channels=2, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, latent_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, 2, L)
        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        z_local = self.pool(h).squeeze(-1)
        return z_local


class AutonomicDecoupledDualStreamModel(nn.Module):
    """
    Proposed Model:
    Combines pre-trained Systemic Autonomic Encoder with Local IV Stream.
    Includes Differential Gating to decouple systemic vasoconstriction from true extravasation.
    """
    def __init__(self, systemic_encoder, num_classes=4, latent_dim=32, freeze_systemic=True):
        super().__init__()
        self.systemic_encoder = systemic_encoder
        if freeze_systemic:
            for p in self.systemic_encoder.parameters():
                p.requires_grad = False

        self.local_encoder = LocalIVEstimator(in_channels=2, latent_dim=latent_dim)

        # Cross-stream projection and differential gating
        self.proj_sys_to_local = nn.Linear(latent_dim, latent_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_local, x_systemic=None):
        """
        Parameters
        ----------
        x_local : torch.Tensor, shape (B, 2, L)
            Local IV transducer signals (temperature, pressure).
        x_systemic : torch.Tensor or None, shape (B, 4, L)
            Systemic wearable signals (TEMP, EDA, BVP, ACC).
            If None (e.g. at runtime with missing systemic sensor), neutral baseline prior is used.
        """
        z_local = self.local_encoder(x_local)

        if x_systemic is not None:
            if len(x_systemic) != len(x_local):
                if len(x_systemic) > len(x_local):
                    x_systemic = x_systemic[:len(x_local)]
                else:
                    repeats = (len(x_local) + len(x_systemic) - 1) // len(x_systemic)
                    x_systemic = x_systemic.repeat(repeats, 1, 1)[:len(x_local)]
            z_sys = self.systemic_encoder(x_systemic)
        else:
            # Zero / neutral physiological baseline
            z_sys = torch.zeros_like(z_local)

        # Expected systemic influence on local features
        expected_sys_effect = self.proj_sys_to_local(z_sys)
        # Residual discrepancy (local minus expected systemic cooling)
        residual_local = z_local - expected_sys_effect

        # Gating weights: determine how much of the local signal is true local variation
        gate = self.gate_net(torch.cat([z_local, z_sys], dim=-1))
        gated_local = gate * residual_local

        # Combined representation
        fused = torch.cat([gated_local, z_sys], dim=-1)
        logits = self.classifier(fused)
        return logits
