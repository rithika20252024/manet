"""
RESILIENT_MANET_STANDALONE/modules/phase3_bayesian.py
=====================================================
Phase 3: Dynamic Trust Calibration with Bayesian Inference
- Model trust as Beta distribution: Beta(alpha, beta) (success / failure counts)
- Implement sequential Bayesian updating with exponential memory decay
- Confidence intervals provide uncertainty estimates for routing decisions
- Adaptive exploration-exploitation trade-off for routing (mu - kappa * sigma)
- Graceful trust poisoning recovery mechanism for rehabilitated nodes
"""

import numpy as np
from scipy.stats import beta as beta_dist
from typing import Dict, List, Tuple
import RESILIENT_MANET_STANDALONE.config as config


class BayesianNodeState:
    """
    Maintains a node's Beta posterior distribution and poisoning recovery state.
    """

    def __init__(self, node_id: int, prior_alpha: float = config.PRIOR_ALPHA,
                 prior_beta: float = config.PRIOR_BETA, decay_lambda: float = config.BAYES_DECAY_LAMBDA):
        self.node_id = node_id
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = prior_alpha
        self.beta = prior_beta
        self.decay_lambda = decay_lambda
        self.consecutive_successes = 0
        self.total_observations = 0

    def sequential_update(self, success_count: int, failure_count: int) -> None:
        """
        Sequential Bayesian update with exponential time discounting:
        alpha(t) = alpha_0 + lambda * (alpha(t-1) - alpha_0) + successes
        beta(t)  = beta_0  + lambda * (beta(t-1)  - beta_0)  + failures
        """
        self.alpha = self.prior_alpha + self.decay_lambda * (self.alpha - self.prior_alpha) + success_count
        self.beta  = self.prior_beta  + self.decay_lambda * (self.beta - self.prior_beta)   + failure_count
        self.total_observations += (success_count + failure_count)

        if failure_count == 0 and success_count > 0:
            self.consecutive_successes += success_count
        else:
            self.consecutive_successes = 0

        # Graceful Poisoning Recovery:
        # If a previously flagged node consistently behaves honestly,
        # apply an accelerated recovery discount to historical failure penalty beta.
        if self.consecutive_successes >= config.RECOVERY_THRESHOLD and self.beta > (self.prior_beta + 2.0):
            self.beta = self.prior_beta + (self.beta - self.prior_beta) * 0.50

    def get_posterior_mean(self) -> float:
        """Expected trust value E[Theta] = alpha / (alpha + beta)."""
        return float(self.alpha / (self.alpha + self.beta))

    def get_variance(self) -> float:
        """Uncertainty measure Var(Theta)."""
        a, b = self.alpha, self.beta
        return float((a * b) / (((a + b) ** 2) * (a + b + 1.0)))

    def get_confidence_interval(self, confidence: float = config.CONFIDENCE_LEVEL) -> Tuple[float, float]:
        """Calculates Bayesian credible interval [lower_bound, upper_bound]."""
        tail = (1.0 - confidence) / 2.0
        low = beta_dist.ppf(tail, self.alpha, self.beta)
        high = beta_dist.ppf(1.0 - tail, self.alpha, self.beta)
        return float(low), float(high)

    def get_risk_adjusted_score(self, kappa: float = config.RISK_PENALTY_KAPPA) -> float:
        """
        Lower confidence bound for adaptive exploration-exploitation routing:
        Score = mu - kappa * sigma
        Favors nodes that are both trustworthy AND have high confidence (low uncertainty).
        """
        mu = self.get_posterior_mean()
        sigma = np.sqrt(self.get_variance())
        return float(np.clip(mu - kappa * sigma, 0.0, 1.0))


class DynamicBayesianTrustCalibrator:
    """
    Manages Bayesian trust calibration across all MANET nodes.
    """

    def __init__(self, n_nodes: int = config.N_NODES):
        self.n_nodes = n_nodes
        self.nodes = {i: BayesianNodeState(i) for i in range(n_nodes)}

    def record_interaction_batch(self, node_id: int, successes: int, failures: int) -> None:
        self.nodes[node_id].sequential_update(successes, failures)

    def get_all_posterior_means(self) -> np.ndarray:
        return np.array([self.nodes[i].get_posterior_mean() for i in range(self.n_nodes)])

    def get_all_risk_adjusted_trust(self) -> np.ndarray:
        return np.array([self.nodes[i].get_risk_adjusted_score() for i in range(self.n_nodes)])

    def get_all_uncertainties(self) -> np.ndarray:
        return np.array([np.sqrt(self.nodes[i].get_variance()) for i in range(self.n_nodes)])
