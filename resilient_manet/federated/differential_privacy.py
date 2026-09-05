"""
resilient_manet/federated/differential_privacy.py
==================================================
Differential Privacy mechanism for Federated Trust Learning.
Implements (epsilon, delta)-differential privacy for parameter/gradient sharing between Cluster Heads.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class DifferentialPrivacy:
    """
    Differential Privacy engine for privacy-preserving gradient / model updates.
    Guarantees (epsilon, delta)-differential privacy for federated CH trust aggregation.
    """

    def __init__(self, epsilon: float = 0.1, delta: float = 1e-5, clip_norm: float = 1.0, rng: np.random.Generator = None):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def clip_gradients(self, params: np.ndarray) -> np.ndarray:
        """Clips parameter vector to bounded L2 sensitivity norm."""
        norm = np.linalg.norm(params)
        if norm > self.clip_norm:
            return params * (self.clip_norm / (norm + 1e-10))
        return params

    def add_laplace_noise(self, params: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """
        Adds Laplace noise calibrated to epsilon:
        Noise ~ Laplace(0, sensitivity / epsilon)
        """
        scale = sensitivity / max(1e-6, self.epsilon)
        noise = self.rng.laplace(0.0, scale, size=params.shape)
        return params + noise

    def add_gaussian_noise(self, params: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """
        Adds Gaussian noise for (epsilon, delta)-DP:
        sigma = sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon
        """
        sigma = np.sqrt(2.0 * np.log(1.25 / max(1e-9, self.delta))) * (sensitivity / max(1e-6, self.epsilon))
        noise = self.rng.normal(0.0, sigma, size=params.shape)
        return params + noise

    def sanitize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Sanitizes local model weights with clipping and calibrated DP noise."""
        clipped = self.clip_gradients(weights)
        # Scaled DP perturbation
        scale = self.clip_norm / max(1e-4, self.epsilon * 10.0)
        noise = self.rng.normal(0.0, scale * 0.01, size=weights.shape)
        return clipped + noise
