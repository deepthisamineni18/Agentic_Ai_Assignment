from __future__ import annotations

import hashlib
import sys
from unittest.mock import MagicMock


class MockSentenceTransformer:
    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name

    def encode(self, sentences: list[str], convert_to_numpy: bool = True):
        import numpy as np

        matrix = np.zeros((len(sentences), 384), dtype=float)
        for i, sentence in enumerate(sentences):
            digest = hashlib.sha256(sentence.encode("utf-8")).digest()
            for j, byte in enumerate(digest):
                matrix[i, j % 384] += float(byte)
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        return matrix


_mock_module = MagicMock()
_mock_module.SentenceTransformer = MockSentenceTransformer
sys.modules["sentence_transformers"] = _mock_module
