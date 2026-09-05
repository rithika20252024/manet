"""
resilient_manet/meta_learning/maml_trust.py
===========================================
Model-Agnostic Meta-Learning (MAML) for Attack Agnosticism
- Meta-trained across task distributions of known and synthetic attack behaviors
- Enables few-shot adaptation to novel/zero-day attacks within 5-10 interaction windows
- Computes inner-loop task-specific gradient update and outer-loop meta-update
"""

import numpy as np
from typing import List, Dict, Tuple


class MAMLTrustModel:
    """Neural trust model parameterized for MAML few-shot adaptation."""

    def __init__(self, in_dim: int = 9, hidden_dim: int = 16, alpha_lr: float = 0.05, beta_lr: float = 0.01, rng: np.random.Generator = None):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.alpha_lr = alpha_lr # inner loop learning rate
        self.beta_lr = beta_lr   # outer loop meta learning rate
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Meta parameters theta
        self.theta_W1 = self.rng.standard_normal((in_dim, hidden_dim)) * 0.1
        self.theta_b1 = np.zeros(hidden_dim)
        self.theta_W2 = self.rng.standard_normal((hidden_dim, 1)) * 0.1
        self.theta_b2 = np.zeros(1)

        # Active adapted parameters
        self.adapted_W1 = self.theta_W1.copy()
        self.adapted_b1 = self.theta_b1.copy()
        self.adapted_W2 = self.theta_W2.copy()
        self.adapted_b2 = self.theta_b2.copy()

    def forward_with_params(self, X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, X @ W1 + b1)
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ W2 + b2, -15.0, 15.0)))
        return p.flatten()

    def compute_gradients(self, X: np.ndarray, y: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
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

    def adapt_few_shot(self, support_X: np.ndarray, support_y: np.ndarray, inner_steps: int = 3) -> None:
        """Fast inner-loop adaptation to novel observed attack patterns."""
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_with_params(X, self.adapted_W1, self.adapted_b1, self.adapted_W2, self.adapted_b2)

    def meta_train_epoch(self, tasks: List[Dict[str, np.ndarray]]) -> float:
        """Executes MAML meta-update across task distribution."""
        meta_grad_W1 = np.zeros_like(self.theta_W1)
        meta_grad_b1 = np.zeros_like(self.theta_b1)
        meta_grad_W2 = np.zeros_like(self.theta_W2)
        meta_grad_b2 = np.zeros_like(self.theta_b2)
        total_loss = 0.0

        for task in tasks:
            sup_X, sup_y = task['support_X'], task['support_y']
            qry_X, qry_y = task['query_X'], task['query_y']

            # Inner step
            gW1, gb1, gW2, gb2 = self.compute_gradients(sup_X, sup_y, self.theta_W1, self.theta_b1, self.theta_W2, self.theta_b2)
            W1_prime = self.theta_W1 - self.alpha_lr * gW1
            b1_prime = self.theta_b1 - self.alpha_lr * gb1
            W2_prime = self.theta_W2 - self.alpha_lr * gW2
            b2_prime = self.theta_b2 - self.alpha_lr * gb2

            # Query gradients for meta-update
            q_gW1, q_gb1, q_gW2, q_gb2 = self.compute_gradients(qry_X, qry_y, W1_prime, b1_prime, W2_prime, b2_prime)
            meta_grad_W1 += q_gW1
            meta_grad_b1 += q_gb1
            meta_grad_W2 += q_gW2
            meta_grad_b2 += q_gb2

            pred_qry = self.forward_with_params(qry_X, W1_prime, b1_prime, W2_prime, b2_prime)
            total_loss += -np.mean(qry_y * np.log(pred_qry + 1e-12) + (1.0 - qry_y) * np.log(1.0 - pred_qry + 1e-12))

        n_tasks = len(tasks)
        self.theta_W1 -= self.beta_lr * (meta_grad_W1 / n_tasks)
        self.theta_b1 -= self.beta_lr * (meta_grad_b1 / n_tasks)
        self.theta_W2 -= self.beta_lr * (meta_grad_W2 / n_tasks)
        self.theta_b2 -= self.beta_lr * (meta_grad_b2 / n_tasks)

        self.adapted_W1 = self.theta_W1.copy()
        self.adapted_b1 = self.theta_b1.copy()
        self.adapted_W2 = self.theta_W2.copy()
        self.adapted_b2 = self.theta_b2.copy()

        return float(total_loss / n_tasks)
