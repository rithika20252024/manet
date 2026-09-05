"""
resilient_manet/bayesian/bayesian_calibration.py
================================================
Dynamic Bayesian Trust Calibration (BTC)
- Treats node trust as a Beta-distribution posterior: Beta(alpha, beta)
- Sequential Bayesian updating: alpha tracks successes, beta tracks failures
- Confidence intervals provide uncertainty bounds [lower_bound, upper_bound]
- Trust poisoning recovery: time-decaying evidence discounting allows rehabilitated nodes to restore credibility gracefully without eternal suspicion penalties
"""

import numpy as np
from scipy.stats import beta as beta_dist
from typing import Dict, List, Tuple, Optional


class BayesianTrustNode:
    """Represents a single node's Bayesian trust state."""

    def __init__(self, node_id: int, prior_alpha: float = 2.0, prior_beta: float = 2.0, decay_lambda: float = 0.95):
        self.node_id = node_id
        self.alpha = prior_alpha
        self.beta = prior_beta
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_lambda = decay_lambda
        self.total_interactions = 0
        self.consecutive_good = 0
        self.poisoning_flag = False

    def update_evidence(self, success_count: int, failure_count: int) -> None:
        """
        Sequential update with graceful exponential discounting:
        alpha(t) = decay * alpha(t-1) + new_successes
        beta(t)  = decay * beta(t-1)  + new_failures
        """
        self.alpha = self.prior_alpha + self.decay_lambda * (self.alpha - self.prior_alpha) + success_count
        self.beta  = self.prior_beta  + self.decay_lambda * (self.beta - self.prior_beta)   + failure_count
        self.total_interactions += (success_count + failure_count)

        if failure_count == 0 and success_count > 0:
            self.consecutive_good += success_count
        else:
            self.consecutive_good = 0

        # Graceful poisoning recovery check: if node demonstrates prolonged consistent honesty
        if self.consecutive_good >= 15 and self.beta > 5.0:
            # Accelerated recovery discount on historical bad penalty
            self.beta = self.prior_beta + (self.beta - self.prior_beta) * 0.5
            self.poisoning_flag = False

    def get_posterior_mean(self) -> float:
        """Expected trust score E[X] = alpha / (alpha + beta)."""
        return float(self.alpha / (self.alpha + self.beta))

    def get_variance(self) -> float:
        """Trust variance representing uncertainty."""
        a, b = self.alpha, self.beta
        return float((a * b) / (((a + b) ** 2) * (a + b + 1.0)))

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Returns Bayesian credible interval [lower, upper]."""
        tail = (1.0 - confidence) / 2.0
        lower = beta_dist.ppf(tail, self.alpha, self.beta)
        upper = beta_dist.ppf(1.0 - tail, self.alpha, self.beta)
        return float(lower), float(upper)

    def get_risk_adjusted_trust(self, penalty_factor: float = 1.5) -> float:
        """
        Lower confidence bound based routing metric:
        score = mu - penalty_factor * sigma
        Prioritizes nodes that are both trustworthy AND have high confidence (low uncertainty).
        """
        mu = self.get_posterior_mean()
        sigma = np.sqrt(self.get_variance())
        return float(np.clip(mu - penalty_factor * sigma, 0.0, 1.0))


class BayesianTrustCalibrator:
    """Manages Bayesian trust calibration for all MANET nodes."""

    def __init__(self, n_nodes: int, decay_lambda: float = 0.95):
        self.n_nodes = n_nodes
        self.nodes = {i: BayesianTrustNode(i, decay_lambda=decay_lambda) for i in range(n_nodes)}

    def record_node_observations(self, node_id: int, successes: int, failures: int) -> None:
        self.nodes[node_id].update_evidence(successes, failures)

    def get_all_trust_scores(self) -> np.ndarray:
        return np.array([self.nodes[i].get_posterior_mean() for i in range(self.n_nodes)])

    def get_all_risk_adjusted_scores(self) -> np.ndarray:
        return np.array([self.nodes[i].get_risk_adjusted_trust() for i in range(self.n_nodes)])

    def get_all_uncertainties(self) -> np.ndarray:
        return np.array([np.sqrt(self.nodes[i].get_variance()) for i in range(self.n_nodes)])
