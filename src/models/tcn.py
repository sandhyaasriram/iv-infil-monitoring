"""
Temporal Convolutional Network (TCN) for IV Infiltration Detection.

Key design decisions:
- Causal convolutions only: no future observations are visible during inference.
  This is enforced by left-padding (causal padding) and zero right-side padding.
- Dilated convolutions: dilation factors [1, 2, 4, 8] give a receptive field of
  ~30 time steps at dilation=8 with kernel_size=3, covering ~30 seconds at 1Hz.
- Residual connections: each TCN block adds a skip connection; the channel-count
  adjustment uses a pointwise 1x1 conv if needed.
- Global average pooling over the time axis aggregates temporal context for
  multi-class window-level classification.

Input:  (B, C, L)  — batch, channels (2=temp+pressure), window length
Output: (B, num_classes) — logits

References:
    Bai et al., 2018, "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modelling" (TCN design).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalDilatedConv1d(nn.Module):
    """Single causal dilated 1D convolution with left-only padding."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        # Left-pad so output length == input length, with NO look-ahead
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,   # We apply padding manually below
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad only on the left (past side) — strictly causal
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class _TCNResidualBlock(nn.Module):
    """
    One TCN residual block: two stacked causal dilated convolutions + BatchNorm
    + GELU activation + residual shortcut.

    Dropout is applied after each activation to regularise the small synthetic dataset.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = _CausalDilatedConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = _CausalDilatedConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Pointwise projection for residual if channel dims differ
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + residual)
        out = self.dropout(out)
        return out


class TCN1D(nn.Module):
    """
    Temporal Convolutional Network for window-level IV infiltration classification.

    Architecture:
        Stem  → Block(dil=1) → Block(dil=2) → Block(dil=4) → Block(dil=8)
              → GlobalAvgPool → FC(64) → FC(num_classes)

    Effective receptive field:
        With kernel_size=3, 4 blocks with dilations [1,2,4,8]:
        RF = 2*(1+2+4+8)*(3-1) + 1 = 61 time steps
        At 1 Hz sampling that equals 61 seconds of look-back.

    Parameters
    ----------
    in_channels : int
        Number of input sensor channels (default=2: temp, pressure).
        Set to 1 for single-sensor ablations.
    num_classes : int
        Number of output classes (4 for none/early/moderate/severe).
    base_channels : int
        Base number of feature maps. Doubles each block pair.
    dropout : float
        Dropout rate applied inside residual blocks.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 4,
                 base_channels: int = 32, dropout: float = 0.15):
        super().__init__()

        ch = base_channels
        self.blocks = nn.Sequential(
            _TCNResidualBlock(in_channels, ch, kernel_size=3, dilation=1, dropout=dropout),
            _TCNResidualBlock(ch, ch, kernel_size=3, dilation=2, dropout=dropout),
            _TCNResidualBlock(ch, ch * 2, kernel_size=3, dilation=4, dropout=dropout),
            _TCNResidualBlock(ch * 2, ch * 2, kernel_size=3, dilation=8, dropout=dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(ch * 2, 64),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, C, L)
        Returns
        -------
        logits : Tensor of shape (B, num_classes)
        """
        out = self.blocks(x)
        out = self.global_pool(out).squeeze(-1)  # (B, ch*2)
        return self.classifier(out)


def build_tcn(in_channels: int = 2, num_classes: int = 4,
              base_channels: int = 32, dropout: float = 0.15) -> TCN1D:
    """Factory function for constructing a TCN1D with standard hyperparameters."""
    return TCN1D(in_channels=in_channels, num_classes=num_classes,
                 base_channels=base_channels, dropout=dropout)
