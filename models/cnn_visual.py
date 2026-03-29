"""CNN Visual Feature Extractor for video frame analysis."""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNVisualExtractor(nn.Module):
    """Extracts visual features from video frames using a pretrained ResNet50 backbone."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC layer
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embedding from a batch of frames. Shape: (B, feature_dim)."""
        with torch.no_grad() if not self.training else torch.enable_grad():
            feat = self.features(x).squeeze(-1).squeeze(-1)
        return self.projector(feat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, features) for a batch of frames."""
        feat = self.features(x).squeeze(-1).squeeze(-1)
        projected = self.projector(feat)
        logits = self.classifier(projected)
        return logits, projected
