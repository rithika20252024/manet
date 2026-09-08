# Technology Stack & Implementation Mapping for RESILIENT-MANET

This document outlines the exact **Technology Stack**, **Libraries**, **Frameworks**, and **Mathematical Algorithms** used to implement every point across all 5 phases of **RESILIENT-MANET**.

---

## 🛠️ Global Technology Stack Summary

| Technology Category | Tool / Framework | Version / Specification | Primary Role in RESILIENT-MANET |
| :--- | :--- | :--- | :--- |
| **Core Runtime** | **Python** | `3.11+ / 3.13` | Execution of all distributed modules, adversarial models, and simulation pipelines |
| **Tensor & Array Math** | **NumPy** | `2.3.2` | Vectorized matrix operations, feature interaction tensors, and gradient updates |
| **Statistical Physics** | **SciPy (scipy.stats)** | `1.16.1` | Beta distributions, PPF quantiles, credible intervals, variance estimation, and paired $t$-tests |
| **Privacy Preservation** | **Differential Privacy Engine** | `(ε=0.1, δ=1e-5)-DP` | Gaussian noise perturbation mechanism and $L_2$-norm gradient clipping |
| **Federated Learning** | **Custom FedAvg Engine** | Distributed CH protocol | Reputation-Weighted FedAvg aggregation across 10 Cluster Heads |
| **Adversarial Deep Learning** | **Custom GATM Neural Engine** | Generator + Discriminator | Online GAN sparring on synthetic + real attack vectors |
| **Meta-Learning** | **Custom MAML Engine** | Bi-Level Optimization | First-order / second-order gradient few-shot adaptation ($\le 5\text{–}10$ windows) |
| **Optimization & Search** | **HLOA Optimization** | 30 agents, 50 iterations | Bio-inspired parameter tuning for 53,249 GNN weight dimensions |
| **Data Logging & Analysis** | **JSON / Standard Logging** | Python native | Structured logging, timestamped tracking, and JSON benchmark storage |

---

## 📌 Phase-by-Phase Technical Mapping

### Phase 1: Federated Trust Learning (FTL) Framework

| Implementation Task | Technology Stack & Tools Used | Implementation Details & Code Reference |
| :--- | :--- | :--- |
| **1. Distributed trust computation module in Python** | **Python 3.11+, NumPy** | [`LocalClusterTrustModel`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase1_ftl.py) running on each Cluster Head. Computes local feed-forward passes ($9 \to 16 \to 1$) over node interaction feature matrices $X_k$ without centralizing raw packet logs. |
| **2. Differential privacy mechanism for gradient sharing** | **NumPy, Mathematical Gaussian Mechanism** | [`DifferentialPrivacyEngine`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase1_ftl.py): Implements $L_2$-norm clipping ($C=1.0$) followed by Gaussian noise addition calibrated to $\epsilon=0.1$ and $\delta=10^{-5}$: <br>$$\sigma = \frac{\sqrt{2\ln(1.25/\delta)} \cdot C}{\epsilon}$$ |
| **3. Secure aggregation protocol** | **NumPy Vectorized FedAvg** | [`aggregate_reputation_fedavg`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase1_ftl.py): Aggregates sanitized updates using **Reputation-Weighted FedAvg**: <br>$$w_k = \frac{N_k \cdot \text{Rep}_k}{\sum_j N_j \cdot \text{Rep}_j}$$ |
| **4. Test on distributed environment with 5–10 cluster heads** | **FCMVC Spatial Partitioning + FTL Manager** | [`FederatedTrustLearningManager`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase1_ftl.py): Simulates 10 distinct Cluster Heads managing local clusters over $1000\,\text{m} \times 1000\,\text{m}$ terrain. |

---

### Phase 2: Generative Adversarial Training (GATM)

| Implementation Task | Technology Stack & Tools Used | Implementation Details & Code Reference |
| :--- | :--- | :--- |
| **1. GAN architecture design for trust classification** | **NumPy Neural Engine, Tanh/Sigmoid/ReLU Activation** | [`GATMGenerator`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase2_gatm.py) ($8\text{D Latent} \to 16 \to 9\text{D Feature}$) and [`GATMDiscriminator`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase2_gatm.py) ($9\text{D Feature} \to 16 \to 1\text{D Binary Trust Probability}$). |
| **2. Generate synthetic attack datasets (incl. novel attack types)** | **NumPy Normal Random Sampling** | `synthesize_attacks()`: Samples $z \sim \mathcal{N}(0, I)$ and generates deceptive profiles (e.g., oscillating trust histories, camouflaged residual energies) to synthesize zero-day attacks. |
| **3. Train discriminator on real + synthetic data** | **Binary Cross-Entropy (BCE) Loss** | Vectorized backpropagation on stacked batches: Real honest ($y=1$), Real confirmed attacks ($y=0$), and Synthetic zero-day attacks ($y=0$). |
| **4. Online adversarial training loop** | **Online Streaming Training Loop** | [`GATMTrustEngine.online_adversarial_step()`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase2_gatm.py): Continuously spars during every time step $t$, hardening the discriminator in real-time. |

---

### Phase 3: Bayesian Trust Calibration (BTC)

| Implementation Task | Technology Stack & Tools Used | Implementation Details & Code Reference |
| :--- | :--- | :--- |
| **1. Model trust as Beta distribution (success/failure counts)** | **`scipy.stats.beta`, Beta Distribution** | [`BayesianNodeState`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase3_bayesian.py): Represents each node's trust as $\text{Beta}(\alpha, \beta)$ with prior parameters $\alpha_0=2.0, \beta_0=2.0$. |
| **2. Sequential Bayesian updating** | **Exponential Time-Discounting Recursion** | `sequential_update()`: Updates parameters sequentially with memory decay factor $\lambda_b = 0.95$: <br>$$\alpha(t) = \alpha_0 + \lambda_b(\alpha(t-1) - \alpha_0) + s(t), \quad \beta(t) = \beta_0 + \lambda_b(\beta(t-1) - \beta_0) + f(t)$$ |
| **3. Confidence-based routing decisions** | **SciPy PPF Quantiles & Risk Formulation** | `get_risk_adjusted_score()`: Computes lower confidence bound $\mu - \kappa \cdot \sigma$ ($\kappa=1.5$), prioritizing nodes that are both trustworthy and certain (low variance). Also includes poisoning recovery when $s \ge 12$. |
| **4. Uncertainty visualization dashboard** | **JSON Structured Export & Matplotlib Engine** | [`outputs/resilient_manet_final_benchmark.json`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/outputs/resilient_manet_final_benchmark.json) exporting uncertainty variance $\sigma^2$ and credible intervals. |

---

### Phase 4: Meta-Learning Integration (MAML)

| Implementation Task | Technology Stack & Tools Used | Implementation Details & Code Reference |
| :--- | :--- | :--- |
| **1. Implement MAML for trust models** | **Custom Bi-Level Gradient Engine (NumPy)** | [`MAMLTrustModel`](file:///Users/rithika/Documents/CODE%202/RESILIENT_MANET_STANDALONE/modules/phase4_maml.py): Implements model-agnostic meta-learning optimizing initial parameters $\theta$. |
| **2. Create task distribution over attack scenarios** | **Multi-Task Sampling Pipeline** | Tasks $\mathcal{T}_i$ generated across 5 distinct attack classes: Blackhole, Grayhole, On-Off, Collusion, and Zero-Day Camouflage attacks. |
| **3. Train meta-model for few-shot adaptation** | **Bi-Level Optimization (Inner LR $\alpha=0.05$, Outer Meta-LR $\beta=0.01$)** | `meta_train_step()`: Computes inner-loop task-specific updates on support sets $\mathcal{S}_i$ and updates initial weights $\theta$ on query sets $\mathcal{Q}_i$. |
| **4. Few-shot adaptation to novel attacks (5–10 windows)** | **`adapt_few_shot()` Engine** | Rapidly adapts weights $\theta \to \theta'$ using only 2–3 gradient steps when unclassified traffic anomalies appear. |

---

### Phase 5: Comprehensive Evaluation & Benchmarking

| Implementation Task | Technology Stack & Tools Used | Implementation Details & Code Reference |
| :--- | :--- | :--- |
| **1. Large-scale simulations (100–200+ nodes)** | **`ResilientMANETSimulationEngine`** | Simulates 100 mobile nodes with Random Waypoint Mobility ($1\text{–}10\,\text{m/s}$), Two-Ray Ground propagation, and CBR UDP traffic over 40s. |
| **2. Real-world testbed & edge emulation** | **Lightweight Memory-Bound Execution** | Low-memory architecture ($\le 128\,\text{MB}$ RAM footprint, $\approx 0.16\text{s}$ per decision), compatible with Raspberry Pi 4 clusters. |
| **3. Compare against baselines** | **`run_all_review2.py` / `main_resilient.py`** | 3-way comparative benchmark comparing Base Paper (*EER-MANET-EFIAGNN*), Senior Extension (*AT-AEES-MANET*), and *SO-RA-MANET*. |
| **4. Key Metrics Tracked** | **SciPy Stats & JSON Aggregator** | Detection Rate ($92.0\text{–}93.0\%$), False Positive Rate ($0.0\%$), End-to-End Delay ($0.068\,\text{ms}$), Throughput ($1304.6\,\text{kbps}$), and Energy Efficiency ($8.58\%$). |
