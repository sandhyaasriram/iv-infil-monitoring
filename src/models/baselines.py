"""
Baseline Architectures for IV Infiltration Monitoring:
1. Logistic Regression (Linear baseline on window features)
2. Random Forest (Ensemble baseline on window features)
3. Shallow 1D-CNN (Standard temporal convolutional baseline)
4. Deep 1D-ResNet (Deep temporal residual neural network baseline)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_logistic_regression_baseline(random_state=42):
    return Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
    ])


def build_random_forest_baseline(random_state=42):
    return Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=random_state))
    ])


class ShallowCNN1D(nn.Module):
    """
    Standard shallow 1D Convolutional Neural Network baseline.
    Input shape: (B, 2, L) -> Output logits: (B, 4)
    """
    def __init__(self, in_channels=2, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


class ResidualBlock1D(nn.Module):
    """1D Residual convolutional building block with optional projection shortcut."""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + residual)
        return out


class DeepResNet1D(nn.Module):
    """
    Deep Residual 1D Temporal Network Baseline (6 convolutional layers).
    Provides a significantly deeper, expressive temporal architecture baseline.
    Input shape: (B, 2, L) -> Output logits: (B, 4)
    """
    def __init__(self, in_channels=2, num_classes=4, base_channels=32):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU()
        )

        # Residual stages
        self.block1 = ResidualBlock1D(base_channels, base_channels, stride=1, dilation=1)
        self.block2 = ResidualBlock1D(base_channels, base_channels * 2, stride=2, dilation=1)
        self.block3 = ResidualBlock1D(base_channels * 2, base_channels * 4, stride=1, dilation=2)

        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(base_channels * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.global_pool(out).squeeze(-1)
        out = self.dropout(out)
        out = F.gelu(self.fc1(out))
        return self.fc2(out)
