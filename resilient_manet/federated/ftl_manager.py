"""
resilient_manet/federated/ftl_manager.py
=======================================
Federated Trust Learning (FTL) Manager
- Distributed trust computation across Cluster Heads (CHs) without central aggregation
- Privacy-preserving gradient sharing with Differential Privacy (epsilon=0.1)
- Local trust models trained on node-specific interaction histories
- Global trust model aggregation using FedAvg with adaptive reputation weighting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from resilient_manet.federated.differential_privacy import DifferentialPrivacy


class LocalTrustModel:
    """Local parameterized trust estimator running on a Cluster Head."""

    def __init__(self, in_dim: int = 9, hidden_dim: int = 16, rng: np.random.Generator = None):
        self.rng = rng if rng is not None else np.random.default_rng(42)
        # Weights for evaluating node risk/trust from features
        self.W1 = self.rng.standard_normal((in_dim, hidden_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.standard_normal((hidden_dim, 1)) * 0.1
        self.b2 = np.zeros(1)
        self.learning_rate = 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass producing trust score [0, 1]."""
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        out = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -15.0, 15.0)))
        return out.flatten()

    def get_flattened_weights(self) -> np.ndarray:
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])

    def set_flattened_weights(self, flat_w: np.ndarray) -> None:
        idx = 0
        w1_size = self.W1.size
        self.W1 = flat_w[idx:idx+w1_size].reshape(self.W1.shape)
        idx += w1_size

        b1_size = self.b1.size
        self.b1 = flat_w[idx:idx+b1_size].reshape(self.b1.shape)
        idx += b1_size

        w2_size = self.W2.size
        self.W2 = flat_w[idx:idx+w2_size].reshape(self.W2.shape)
        idx += w2_size

        b2_size = self.b2.size
        self.b2 = flat_w[idx:idx+b2_size].reshape(self.b2.shape)

    def train_step(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> float:
        """Local training on cluster interaction history."""
        if len(X) == 0:
            return 0.0
        total_loss = 0.0
        for _ in range(epochs):
            # Forward
            h = np.maximum(0.0, X @ self.W1 + self.b1)
            p = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W2 + self.b2, -15.0, 15.0))).flatten()
            
            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))
            total_loss += loss

            # Gradients
            grad_out = (p - y)[:, np.newaxis] / len(X)
            grad_W2 = h.T @ grad_out
            grad_b2 = np.sum(grad_out, axis=0)

            grad_h = grad_out @ self.W2.T * (h > 0)
            grad_W1 = X.T @ grad_h
            grad_b1 = np.sum(grad_h, axis=0)

            # Update
            self.W1 -= self.learning_rate * grad_W1
            self.b1 -= self.learning_rate * grad_b1
            self.W2 -= self.learning_rate * grad_W2
            self.b2 -= self.learning_rate * grad_b2

        return total_loss / epochs


class FTLClusterManager:
    """
    Federated Trust Learning Coordinator across all Cluster Heads.
    Computes distributed model updates without transmitting raw node logs.
    """

    def __init__(self, n_clusters: int, in_dim: int = 9, epsilon: float = 0.1, rng: np.random.Generator = None):
        self.n_clusters = n_clusters
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.dp = DifferentialPrivacy(epsilon=epsilon, rng=self.rng)
        self.global_model = LocalTrustModel(in_dim=in_dim, rng=self.rng)
        self.local_models = [LocalTrustModel(in_dim=in_dim, rng=self.rng) for _ in range(n_clusters)]
        self.reputations = np.ones(n_clusters)

    def distribute_global_model(self) -> None:
        """Sends current global model weights to all cluster heads."""
        global_w = self.global_model.get_flattened_weights()
        for lm in self.local_models:
            lm.set_flattened_weights(global_w.copy())

    def local_update(self, cluster_idx: int, X_local: np.ndarray, y_local: np.ndarray) -> np.ndarray:
        """Trains local model and returns sanitized (DP-perturbed) weights."""
        lm = self.local_models[cluster_idx]
        lm.train_step(X_local, y_local, epochs=5)
        local_w = lm.get_flattened_weights()
        sanitized_w = self.dp.sanitize_weights(local_w)
        return sanitized_w

    def aggregate_fedavg(self, sanitized_weights_list: List[np.ndarray], sample_counts: List[int], ch_trust_scores: List[float]) -> None:
        """
        Aggregates DP sanitized updates with Reputation-Weighted FedAvg:
        weight_k = (n_k * reputation_k) / sum(n_j * reputation_j)
        """
        if not sanitized_weights_list:
            return

        total_score = 0.0
        weights = []
        for n_k, rep in zip(sample_counts, ch_trust_scores):
            score = max(1, n_k) * max(0.1, rep)
            weights.append(score)
            total_score += score

        weights = np.array(weights) / total_score

        aggregated_w = np.zeros_like(sanitized_weights_list[0])
        for w_k, alpha_k in zip(sanitized_weights_list, weights):
            aggregated_w += alpha_k * w_k

        self.global_model.set_flattened_weights(aggregated_w)
        self.distribute_global_model()

    def predict_trust(self, features: np.ndarray) -> np.ndarray:
        """Evaluates node feature vectors using the global federated trust model."""
        return self.global_model.forward(features)
