"""OCR Text Detection module for extracting text features from video frames."""

import torch
import torch.nn as nn
import numpy as np

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


class OCRTextDetector(nn.Module):
    """Detects and encodes text from video frames into a fixed-size feature vector."""

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, feature_dim: int = 128, max_tokens: int = 32):
        super().__init__()
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * max_tokens, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim),
        )
        self.classifier_head = nn.Linear(feature_dim, 10)
        self._char_to_idx: dict[str, int] = {}

    def _build_vocab(self, text: str) -> list[int]:
        """Simple character-level tokenization."""
        tokens = []
        for ch in text.lower()[:self.max_tokens]:
            if ch not in self._char_to_idx:
                idx = len(self._char_to_idx) + 1  # 0 is padding
                if idx < self.vocab_size:
                    self._char_to_idx[ch] = idx
            tokens.append(self._char_to_idx.get(ch, 1))
        # Pad to max_tokens
        tokens = tokens + [0] * (self.max_tokens - len(tokens))
        return tokens

    def ocr_extract(self, frames_np: np.ndarray) -> list[str]:
        """Run OCR on numpy frames (B, H, W, C) uint8. Returns list of extracted strings."""
        texts = []
        for frame in frames_np:
            if HAS_TESSERACT:
                img = Image.fromarray(frame)
                text = pytesseract.image_to_string(img, timeout=2)
            else:
                text = ""
            texts.append(text.strip())
        return texts

    def texts_to_tensor(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Convert raw text strings to token tensor (B, max_tokens)."""
        batch_tokens = [self._build_vocab(t) for t in texts]
        return torch.tensor(batch_tokens, dtype=torch.long, device=device)

    def extract_features(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs into feature vectors. Input: (B, max_tokens) -> (B, feature_dim)."""
        embedded = self.embedding(token_ids)  # (B, max_tokens, embed_dim)
        flat = embedded.view(embedded.size(0), -1)  # (B, max_tokens * embed_dim)
        return self.encoder(flat)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, features) from token IDs."""
        features = self.extract_features(token_ids)
        logits = self.classifier_head(features)
        return logits, features
