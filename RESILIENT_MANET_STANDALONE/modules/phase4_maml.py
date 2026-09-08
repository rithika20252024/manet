"""
RESILIENT_MANET_STANDALONE/modules/phase4_maml.py
=================================================
Phase 4: Meta-Learning for Attack Agnosticism
- Implement MAML (Model-Agnostic Meta-Learning) for neural trust models
- Create task distribution over diverse attack scenarios (blackhole, grayhole, on-off, collusion, zero-day)
- Train meta-model for rapid few-shot adaptation within 5-10 interaction windows
- Evaluate generalization on unseen/zero-day attack types
"""

import numpy as np
from typing import List, Dict, Tuple
import RESILIENT_MANET_STANDALONE.config as config


class MAMLTrustModel:
    """
    MAML-parameterized neural trust estimator.
    Optimizes initial parameters theta for fast gradient-based adaptation to novel attack dynamics.
    """

    def __init__(self, in_dim: int = 9, hidden_dim: int = 16,
                 alpha_lr: float = config.MAML_ALPHA_LR, beta_lr: float = config.MAML_BETA_LR,
                 rng: np.random.Generator = None):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.alpha_lr = alpha_lr # Inner loop step size
        self.beta_lr = beta_lr   # Outer loop meta-step size
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Meta-parameters Theta
        self.theta_W1 = self.rng.standard_normal((in_dim, hidden_dim)) * 0.1
        self.theta_b1 = np.zeros(hidden_dim)
        self.theta_W2 = self.rng.standard_normal((hidden_dim, 1)) * 0.1
        self.theta_b2 = np.zeros(1)

        # Active adapted parameters
        self.adapted_W1 = self.theta_W1.copy()
        self.adapted_b1 = self.theta_b1.copy()
        self.adapted_W2 = self.theta_W2.copy()
        self.adapted_b2 = self.theta_b2.copy()

    def forward(self, X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, X @ W1 + b1)
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ W2 + b2, -15.0, 15.0)))
        return p.flatten()

    def compute_gradients(self, X: np.ndarray, y: np.ndarray,
                          W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
        if len(X) == 0:
            return np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)
        h = np.maximum(0.0, X @ W1 + b1)
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ W2 + b2, -15.0, 15.0))).flatten()

        grad_out = (p - y)[:, np.newaxis] / len(X)
        grad_W2 = h.T @ grad_out
        grad_b2 = np.sum(grad_out, axis=0)

        grad_h = grad_out @ W2.T * (h > 0)
        grad_W1 = X.T @ grad_h
        grad_b1 = np.sum(grad_h, axis=0)

        return grad_W1, grad_b1, grad_W2, grad_b2

    def adapt_few_shot(self, support_X: np.ndarray, support_y: np.ndarray,
                       inner_steps: int = config.MAML_INNER_STEPS) -> None:
        """Fast adaptation to newly observed/flagged node interaction windows."""
        W1 = self.theta_W1.copy()
        b1 = self.theta_b1.copy()
        W2 = self.theta_W2.copy()
        b2 = self.theta_b2.copy()

        for _ in range(inner_steps):
            gW1, gb1, gW2, gb2 = self.compute_gradients(support_X, support_y, W1, b1, W2, b2)
            W1 -= self.alpha_lr * gW1
            b1 -= self.alpha_lr * gb1
            W2 -= self.alpha_lr * gW2
            b2 -= self.alpha_lr * gb2

        self.adapted_W1 = W1
        self.adapted_b1 = b1
        self.adapted_W2 = W2
        self.adapted_b2 = b2

    def meta_train_step(self, task_batch: List[Dict[str, np.ndarray]]) -> float:
        """Executes outer-loop meta-gradient update across task distributions."""
        meta_gW1 = np.zeros_like(self.theta_W1)
        meta_gb1 = np.zeros_like(self.theta_b1)
        meta_gW2 = np.zeros_like(self.theta_W2)
        meta_gb2 = np.zeros_like(self.theta_b2)
        total_loss = 0.0

        for task in task_batch:
            sX, sy = task['support_X'], task['support_y']
            qX, qy = task['query_X'], task['query_y']

            # Inner step
            gW1, gb1, gW2, gb2 = self.compute_gradients(sX, sy, self.theta_W1, self.theta_b1, self.theta_W2, self.theta_b2)
            W1_prime = self.theta_W1 - self.alpha_lr * gW1
            b1_prime = self.theta_b1 - self.alpha_lr * gb1
            W2_prime = self.theta_W2 - self.alpha_lr * gW2
            b2_prime = self.theta_b2 - self.alpha_lr * gb2

            # Query evaluation
            q_gW1, q_gb1, q_gW2, q_gb2 = self.compute_gradients(qX, qy, W1_prime, b1_prime, W2_prime, b2_prime)
            meta_gW1 += q_gW1
            meta_gb1 += q_gb1
            meta_gW2 += q_gW2
            meta_gb2 += q_gb2

            pred = self.forward(qX, W1_prime, b1_prime, W2_prime, b2_prime)
            total_loss += -np.mean(qy * np.log(pred + 1e-12) + (1.0 - qy) * np.log(1.0 - pred + 1e-12))

        N = len(task_batch)
        self.theta_W1 -= self.beta_lr * (meta_gW1 / N)
        self.theta_b1 -= self.beta_lr * (meta_gb1 / N)
        self.theta_W2 -= self.beta_lr * (meta_gW2 / N)
        self.theta_b2 -= self.beta_lr * (meta_gb2 / N)

        self.adapted_W1 = self.theta_W1.copy()
        self.adapted_b1 = self.theta_b1.copy()
        self.adapted_W2 = self.theta_W2.copy()
        self.adapted_b2 = self.theta_b2.copy()

        return float(total_loss / N)

    def predict_trust(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, self.adapted_W1, self.adapted_b1, self.adapted_W2, self.adapted_b2)
