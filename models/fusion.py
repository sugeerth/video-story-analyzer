"""Learned Fusion Layer and Multi-Modal Ensemble for combining CNN + LSTM + OCR."""

import torch
import torch.nn as nn


class LearnedFusionLayer(nn.Module):
    """Learns optimal weighting and non-linear combination of multi-modal features."""

    def __init__(self, cnn_dim: int = 256, lstm_dim: int = 256, ocr_dim: int = 128, fused_dim: int = 256):
        super().__init__()
        total_dim = cnn_dim + lstm_dim + ocr_dim

        # Gated attention: learn per-modality importance
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid(),
        )

        # Non-linear fusion
        self.fuse = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, fused_dim),
            nn.ReLU(),
        )

    def forward(self, cnn_feat: torch.Tensor, lstm_feat: torch.Tensor, ocr_feat: torch.Tensor) -> torch.Tensor:
        """Fuse three modality feature vectors. Each input: (B, dim). Output: (B, fused_dim)."""
        concat = torch.cat([cnn_feat, lstm_feat, ocr_feat], dim=-1)
        gated = concat * self.gate(concat)
        return self.fuse(gated)


class MultiModalEnsemble(nn.Module):
    """Full ensemble: CNN (visual) + LSTM (temporal) + OCR (text) with learned fusion.

    Target performance: F1 82.2% on logo detection benchmark.
    Designed for production deployment serving millions of devices.
    """

    def __init__(self, num_classes: int = 10, feature_dim: int = 256):
        super().__init__()
        from .cnn_visual import CNNVisualExtractor
        from .lstm_temporal import LSTMTemporal
        from .ocr_text import OCRTextDetector

        self.cnn = CNNVisualExtractor(num_classes=num_classes, feature_dim=feature_dim)
        self.lstm = LSTMTemporal(input_dim=feature_dim, hidden_dim=128, num_classes=num_classes)
        self.ocr = OCRTextDetector(feature_dim=128)

        lstm_out = 128 * 2  # bidirectional
        self.fusion = LearnedFusionLayer(cnn_dim=feature_dim, lstm_dim=lstm_out, ocr_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        frames: torch.Tensor,
        ocr_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            frames: (B, T, C, H, W) video frames
            ocr_tokens: (B, max_tokens) OCR token IDs

        Returns:
            dict with 'logits', 'cnn_logits', 'lstm_logits', 'ocr_logits', 'fused_features'
        """
        B, T, C, H, W = frames.shape

        # CNN: process each frame, average pool across time
        frames_flat = frames.view(B * T, C, H, W)
        cnn_logits, cnn_feats = self.cnn(frames_flat)
        cnn_feats = cnn_feats.view(B, T, -1)  # (B, T, feature_dim)
        cnn_pooled = cnn_feats.mean(dim=1)  # (B, feature_dim)
        cnn_logits = cnn_logits.view(B, T, -1).mean(dim=1)

        # LSTM: temporal modeling over frame features
        lstm_logits, lstm_feats = self.lstm(cnn_feats)  # input: (B, T, feature_dim)

        # OCR: text features
        ocr_logits, ocr_feats = self.ocr(ocr_tokens)

        # Fusion
        fused = self.fusion(cnn_pooled, lstm_feats, ocr_feats)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "cnn_logits": cnn_logits,
            "lstm_logits": lstm_logits,
            "ocr_logits": ocr_logits,
            "fused_features": fused,
        }
