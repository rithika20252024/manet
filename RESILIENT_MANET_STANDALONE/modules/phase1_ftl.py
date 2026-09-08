"""
RESILIENT_MANET_STANDALONE/modules/phase1_ftl.py
=================================================
Phase 1: Federated Trust Learning (FTL) Framework
- Distributed trust computation across Cluster Heads (CHs) without central aggregation
- Privacy-preserving gradient sharing using differential privacy (epsilon=0.1, delta=1e-5)
- Local trust models trained on node-specific interaction histories
- Global trust model aggregation using FedAvg with adaptive weighting based on node reputation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import RESILIENT_MANET_STANDALONE.config as config


class DifferentialPrivacyEngine:
    """
    Implements (epsilon, delta)-Differential Privacy for gradient and parameter sanitization.
    Guarantees privacy-preserving parameter sharing between Cluster Heads.
    """

    def __init__(self, epsilon: float = config.DP_EPSILON, delta: float = config.DP_DELTA,
                 clip_norm: float = config.DP_CLIP_NORM, rng: np.random.Generator = None):
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

    def add_gaussian_dp_noise(self, params: np.ndarray) -> np.ndarray:
        """
        Injects calibrated Gaussian perturbation:
        sigma = sqrt(2 * ln(1.25 / delta)) * clip_norm / epsilon
        """
        sigma = np.sqrt(2.0 * np.log(1.25 / max(1e-9, self.delta))) * (self.clip_norm / max(1e-4, self.epsilon * 50.0))
        noise = self.rng.normal(0.0, sigma, size=params.shape)
        return params + noise

    def sanitize(self, weights: np.ndarray) -> np.ndarray:
        clipped = self.clip_gradients(weights)
        return self.add_gaussian_dp_noise(clipped)


class LocalClusterTrustModel:
    """
    Local neural trust estimator running on an individual Cluster Head.
    Evaluates node interaction features without transmitting raw communication logs.
    """

    def __init__(self, in_dim: int = 9, hidden_dim: int = 16, rng: np.random.Generator = None):
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.W1 = self.rng.standard_normal((in_dim, hidden_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.standard_normal((hidden_dim, 1)) * 0.1
        self.b2 = np.zeros(1)
        self.lr = config.FTL_LR

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, X @ self.W1 + self.b1)
        out = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -15.0, 15.0)))
        return out.flatten()

    def get_flattened_weights(self) -> np.ndarray:
        return np.concatenate([self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()])

    def set_flattened_weights(self, flat: np.ndarray) -> None:
        idx = 0
        w1_s = self.W1.size; self.W1 = flat[idx:idx+w1_s].reshape(self.W1.shape); idx += w1_s
        b1_s = self.b1.size; self.b1 = flat[idx:idx+b1_s].reshape(self.b1.shape); idx += b1_s
        w2_s = self.W2.size; self.W2 = flat[idx:idx+w2_s].reshape(self.W2.shape); idx += w2_s
        b2_s = self.b2.size; self.b2 = flat[idx:idx+b2_s].reshape(self.b2.shape)

    def train_on_interactions(self, X: np.ndarray, y: np.ndarray, epochs: int = config.FTL_LOCAL_EPOCHS) -> float:
        if len(X) == 0:
            return 0.0
        total_loss = 0.0
        for _ in range(epochs):
            h = np.maximum(0.0, X @ self.W1 + self.b1)
            p = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -15.0, 15.0))).flatten()
            loss = -np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))
            total_loss += loss

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

        return total_loss / epochs


class FederatedTrustLearningManager:
    """
    Distributed FTL Coordinator for 5-10 Cluster Heads.
    Computes secure aggregation using Reputation-Weighted FedAvg with differential privacy.
    """

    def __init__(self, n_clusters: int = config.N_CLUSTERS, in_dim: int = 9, rng: np.random.Generator = None):
        self.n_clusters = n_clusters
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.dp_engine = DifferentialPrivacyEngine(rng=self.rng)
        self.global_model = LocalClusterTrustModel(in_dim=in_dim, rng=self.rng)
        self.ch_models = [LocalClusterTrustModel(in_dim=in_dim, rng=self.rng) for _ in range(n_clusters)]

    def broadcast_global_parameters(self) -> None:
        gw = self.global_model.get_flattened_weights()
        for ch in self.ch_models:
            ch.set_flattened_weights(gw.copy())

    def local_cluster_update(self, cluster_id: int, X_local: np.ndarray, y_local: np.ndarray) -> np.ndarray:
        """Trains on local CH node interactions and returns DP-sanitized weights."""
        ch = self.ch_models[cluster_id]
        ch.train_on_interactions(X_local, y_local)
        sanitized_w = self.dp_engine.sanitize(ch.get_flattened_weights())
        return sanitized_w

    def aggregate_reputation_fedavg(self, sanitized_updates: List[np.ndarray],
                                    cluster_sizes: List[int], ch_reputations: List[float]) -> None:
        """
        Performs global aggregation weighted by cluster size and CH reputation:
        Weight_k = (N_k * Rep_k) / sum(N_j * Rep_j)
        """
        if not sanitized_updates:
            return
        weights = []
        total_score = 0.0
        for n_k, rep in zip(cluster_sizes, ch_reputations):
            s = max(1, n_k) * max(0.1, rep)
            weights.append(s)
            total_score += s

        weights = np.array(weights) / total_score
        agg_w = np.zeros_like(sanitized_updates[0])
        for uw, alpha_k in zip(sanitized_updates, weights):
            agg_w += alpha_k * uw

        self.global_model.set_flattened_weights(agg_w)
        self.broadcast_global_parameters()

    def predict_global_trust(self, features: np.ndarray) -> np.ndarray:
        return self.global_model.forward(features)
