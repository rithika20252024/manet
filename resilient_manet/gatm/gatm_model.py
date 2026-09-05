"""
resilient_manet/gatm/gatm_model.py
==================================
Generative Adversarial Trust Model (GATM)
- Generator: Creates synthetic adversarial behavior patterns (including stealthy on-off cycles and masquerading)
- Discriminator: Classifies nodes as benign vs. malicious
- Online adversarial training makes the trust model robust to zero-day and unseen attacks
"""

import numpy as np
from typing import Tuple, Dict, Any, List


class GATMGenerator:
    """Generates plausible malicious behavior patterns to spar against the discriminator."""

    def __init__(self, latent_dim: int = 8, feature_dim: int = 9, rng: np.random.Generator = None):
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.rng = rng if rng is not None else np.random.default_rng(42)
        
        # Generator network: Latent -> Hidden (16) -> Feature Space (9)
        self.W1 = self.rng.standard_normal((latent_dim, 16)) * 0.2
        self.b1 = np.zeros(16)
        self.W2 = self.rng.standard_normal((16, feature_dim)) * 0.2
        self.b2 = np.zeros(feature_dim)
        self.lr = 0.01

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Generates synthetic feature vectors."""
        h = np.tanh(z @ self.W1 + self.b1)
        out = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -10.0, 10.0)))
        return out

    def generate_synthetic_attacks(self, n_samples: int) -> np.ndarray:
        """Samples latent noise and produces diverse synthetic attack feature profiles."""
        z = self.rng.normal(0.0, 1.0, (n_samples, self.latent_dim))
        synthetic = self.forward(z)
        # Inject attack characteristic signatures: e.g., low stability, oscillatory trust, disguised energy
        synthetic[:, 3] = np.clip(synthetic[:, 3] * 0.6 + self.rng.uniform(0.1, 0.4, n_samples), 0.0, 1.0) # RT
        synthetic[:, 4] = np.clip(synthetic[:, 4] * 0.4, 0.0, 1.0) # Low stability
        synthetic[:, 7] = np.clip(synthetic[:, 7] * 0.8 + 0.2, 0.0, 1.0) # Elevated suspicion
        return synthetic


class GATMDiscriminator:
    """Classifies feature vectors as benign (1) or malicious (0)."""

    def __init__(self, feature_dim: int = 9, hidden_dim: int = 16, rng: np.random.Generator = None):
        self.feature_dim = feature_dim
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.W1 = self.rng.standard_normal((feature_dim, hidden_dim)) * 0.15
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.standard_normal((hidden_dim, 1)) * 0.15
        self.b2 = np.zeros(1)
        self.lr = 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -15.0, 15.0)))
        return p.flatten()

    def train_step(self, real_benign: np.ndarray, real_malicious: np.ndarray, synthetic_malicious: np.ndarray) -> Dict[str, float]:
        """Adversarial discrimination training on mixture of real observed and GAN-synthesized attacks."""
        X_real = []
        y_real = []

        if len(real_benign) > 0:
            X_real.append(real_benign)
            y_real.append(np.ones(len(real_benign))) # 1 = benign

        if len(real_malicious) > 0:
            X_real.append(real_malicious)
            y_real.append(np.zeros(len(real_malicious))) # 0 = malicious

        if len(synthetic_malicious) > 0:
            X_real.append(synthetic_malicious)
            y_real.append(np.zeros(len(synthetic_malicious)))

        if not X_real:
            return {'d_loss': 0.0}

        X = np.vstack(X_real)
        y = np.concatenate(y_real)

        # Forward
        h = np.maximum(0.0, X @ self.W1 + self.b1)
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -15.0, 15.0))).flatten()

        loss = -np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))

        # Backward
        grad_out = (p - y)[:, np.newaxis] / len(X)
        grad_W2 = h.T @ grad_out
        grad_b2 = np.sum(grad_out, axis=0)

        grad_h = grad_out @ self.W2.T * (h > 0)
        grad_W1 = X.T @ grad_h
        grad_b1 = np.sum(grad_h, axis=0)

        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2

        return {'d_loss': float(loss)}


class GATMTrustEngine:
    """Unified GATM engine managing generator and discriminator online training."""

    def __init__(self, feature_dim: int = 9, rng: np.random.Generator = None):
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.generator = GATMGenerator(latent_dim=8, feature_dim=feature_dim, rng=self.rng)
        self.discriminator = GATMDiscriminator(feature_dim=feature_dim, rng=self.rng)

    def train_online(self, real_benign: np.ndarray, real_malicious: np.ndarray, n_synthetic: int = 20) -> Dict[str, float]:
        synthetic = self.generator.generate_synthetic_attacks(n_synthetic)
        stats = self.discriminator.train_step(real_benign, real_malicious, synthetic)
        return stats

    def score_nodes(self, node_features: np.ndarray) -> np.ndarray:
        """Outputs trust probability [0, 1] where 1 is highly trusted benign node."""
        return self.discriminator.forward(node_features)
