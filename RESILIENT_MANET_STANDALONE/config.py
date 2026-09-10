"""
RESILIENT_MANET_STANDALONE/config.py
====================================
Configuration parameters for RESILIENT-MANET:
Robust and Energy-Efficient Secure Routing in MANETs with
Federated Adversarial Learning and Dynamic Trust Calibration
"""

# --- Network & Simulation Topology ---
N_NODES             = 100           # Simulation scale (evaluable up to 200+)
SIM_TIME            = 40.0          # Total simulation time (seconds)
AREA_SIZE           = 1000.0        # 1000m x 1000m terrain
E_INITIAL           = 15.1          # Initial energy in Joules per node
E_TX                = 50e-9         # 50 nJ/bit transmit energy
E_RX                = 50e-9         # 50 nJ/bit receive energy
E_AMP               = 100e-12       # 100 pJ/bit/m^2 channel amplification
PKT_SIZE            = 512           # 512 bytes per CBR packet
DATA_RATE           = 4             # 4 packets/second
TX_RANGE            = 250.0         # 250 metres transmission range
MIN_SPEED           = 1.0           # 1 m/s
MAX_SPEED           = 10.0          # 10 m/s
PAUSE_TIME          = 5.0           # 5 seconds pause time
N_RUNS              = 40            # 40 Monte Carlo runs
SEED                = 42            # Master reproducible seed

# --- Adversary & Multi-Attack Model ---
MALICIOUS_RATIO     = 0.10          # 10% malicious nodes
ATTACK_TYPES        = {
    'blackhole' : 0.25,             # 100% unconditional packet dropping
    'grayhole'  : 0.25,             # 60% selective packet dropping
    'on_off'    : 0.25,             # 10s honest / 10s malicious duty cycle
    'collusion' : 0.15,             # Colluding reputation inflation & 75% dropping
    'zero_day'  : 0.10              # Unseen stealthy camouflage zero-day attack
}

# --- Phase 1: Federated Trust Learning (FTL) ---
N_CLUSTERS          = 10            # 10 Cluster Heads for distributed FTL
DP_EPSILON          = 0.1           # Privacy budget epsilon = 0.1
DP_DELTA            = 1e-5          # Differential privacy delta bound
DP_CLIP_NORM        = 1.0           # L2 norm clipping threshold
FTL_LOCAL_EPOCHS    = 5             # Local training epochs per CH
FTL_LR              = 0.01          # Local learning rate

# --- Phase 2: Generative Adversarial Trust Model (GATM) ---
GATM_LATENT_DIM     = 8             # Latent noise dimension for generator
GATM_FEATURE_DIM    = 9             # Node interaction feature space dimension
GATM_HIDDEN_DIM     = 16            # Generator/Discriminator hidden units
GATM_SYNTHETIC_NUM  = 25            # Number of synthetic zero-day attacks generated per round
GATM_LR             = 0.01          # Online adversarial learning rate

# --- Phase 3: Dynamic Bayesian Trust Calibration (BTC) ---
PRIOR_ALPHA         = 2.0           # Beta prior alpha (initial honest evidence)
PRIOR_BETA          = 2.0           # Beta prior beta (initial malicious evidence)
BAYES_DECAY_LAMBDA  = 0.95          # Exponential time discounting factor
CONFIDENCE_LEVEL    = 0.95          # 95% Bayesian credible interval
RISK_PENALTY_KAPPA  = 1.5           # Exploration-exploitation trade-off parameter (mu - kappa * sigma)
RECOVERY_THRESHOLD  = 12            # Consecutive honest forwards for poisoning recovery

# --- Phase 4: Model-Agnostic Meta-Learning (MAML) ---
MAML_ALPHA_LR       = 0.05          # Inner loop adaptation step size
MAML_BETA_LR        = 0.01          # Outer loop meta-gradient step size
MAML_INNER_STEPS    = 3             # Number of few-shot inner adaptation steps
MAML_ADAPT_WINDOWS  = 5             # Fast adaptation demonstrated within 5-10 windows

# --- Phase 5: Routing & Evaluation ---
EVAL_TIMES          = [10.0, 30.0, 40.0] # Evaluation checkpoints
BASELINES           = ['EER-MANET-EFIAGNN', 'AT-AEES-MANET', 'SO-RA-MANET']
