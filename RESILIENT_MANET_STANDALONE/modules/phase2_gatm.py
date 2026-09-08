"""
RESILIENT_MANET_STANDALONE/modules/phase2_gatm.py
==================================================
Phase 2: Generative Adversarial Trust Model (GATM)
- Generator: Creates synthetic malicious behavior patterns including novel zero-day & stealthy on-off cycles
- Discriminator: Classifies nodes as benign vs. malicious
- Adversarial training makes the model robust to unseen attack types
- Online learning capability: Model updates continuously with new streaming observations
"""

import numpy as np
from typing import Dict, List, Tuple
import RESILIENT_MANET_STANDALONE.config as config


class GATMGenerator:
    """
    Generator network G(z): Maps latent noise into realistic deceptive attack profiles.
    Synthesizes zero-day attack features that evade sliding-window rules.
    """

    def __init__(self, latent_dim: int = config.GATM_LATENT_DIM,
                 feature_dim: int = config.GATM_FEATURE_DIM, rng: np.random.Generator = None):
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.rng = rng if rng is not None else np.random.default_rng(42)

        self.W1 = self.rng.standard_normal((latent_dim, config.GATM_HIDDEN_DIM)) * 0.2
        self.b1 = np.zeros(config.GATM_HIDDEN_DIM)
        self.W2 = self.rng.standard_normal((config.GATM_HIDDEN_DIM, feature_dim)) * 0.2
        self.b2 = np.zeros(feature_dim)
        self.lr = config.GATM_LR

    def forward(self, z: np.ndarray) -> np.ndarray:
        h = np.tanh(z @ self.W1 + self.b1)
        out = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -10.0, 10.0)))
        return out

    def synthesize_attacks(self, n_samples: int = config.GATM_SYNTHETIC_NUM) -> np.ndarray:
        z = self.rng.normal(0.0, 1.0, (n_samples, self.latent_dim))
        synthetic = self.forward(z)
        # Inject realistic adversarial properties (intermittent drops, disguise metrics)
        synthetic[:, 3] = np.clip(synthetic[:, 3] * 0.65 + self.rng.uniform(0.1, 0.35, n_samples), 0.0, 1.0)
        synthetic[:, 4] = np.clip(synthetic[:, 4] * 0.35, 0.0, 1.0) # low stability
        synthetic[:, 7] = np.clip(synthetic[:, 7] * 0.8 + 0.2, 0.0, 1.0) # elevated suspicion
        return synthetic


class GATMDiscriminator:
    """
    Discriminator network D(x): Classifies whether a node profile is honest (1) or malicious (0).
    """

    def __init__(self, feature_dim: int = config.GATM_FEATURE_DIM,
                 hidden_dim: int = config.GATM_HIDDEN_DIM, rng: np.random.Generator = None):
        self.feature_dim = feature_dim
        self.rng = rng if rng is not None else np.random.default_rng(42)

        self.W1 = self.rng.standard_normal((feature_dim, hidden_dim)) * 0.15
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.standard_normal((hidden_dim, 1)) * 0.15
        self.b2 = np.zeros(1)
        self.lr = config.GATM_LR

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, X @ self.W1 + self.b1)
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -15.0, 15.0)))
        return p.flatten()

    def train_step(self, real_benign: np.ndarray, real_malicious: np.ndarray,
                   synthetic_malicious: np.ndarray) -> float:
        X_list = []
        y_list = []

        if len(real_benign) > 0:
            X_list.append(real_benign)
            y_list.append(np.ones(len(real_benign))) # 1 = honest

        if len(real_malicious) > 0:
            X_list.append(real_malicious)
            y_list.append(np.zeros(len(real_malicious))) # 0 = malicious

        if len(synthetic_malicious) > 0:
            X_list.append(synthetic_malicious)
            y_list.append(np.zeros(len(synthetic_malicious))) # 0 = synthetic zero-day

        if not X_list:
            return 0.0

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        h = np.maximum(0.0, X @ self.W1 + self.b1)
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -15.0, 15.0))).flatten()

        loss = -np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))

        # Backpropagation
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

        return float(loss)


class GATMTrustEngine:
    """
    Unified GATM Engine managing continuous online adversarial training loop.
    """

    def __init__(self, feature_dim: int = 9, rng: np.random.Generator = None):
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.generator = GATMGenerator(feature_dim=feature_dim, rng=self.rng)
        self.discriminator = GATMDiscriminator(feature_dim=feature_dim, rng=self.rng)

    def online_adversarial_step(self, real_benign: np.ndarray, real_malicious: np.ndarray) -> float:
        synthetic = self.generator.synthesize_attacks(config.GATM_SYNTHETIC_NUM)
        loss = self.discriminator.train_step(real_benign, real_malicious, synthetic)
        return loss

    def evaluate_trust_probabilities(self, features: np.ndarray) -> np.ndarray:
        return self.discriminator.forward(features)
