# RESILIENT-MANET: Standalone Architecture & Experimental Benchmark
## Robust and Energy-Efficient Secure Routing in MANETs with Federated Adversarial Learning and Dynamic Trust Calibration

---

### Project Overview
**RESILIENT-MANET** is a standalone decentralized framework developed from scratch to resolve fundamental security, privacy, and adaptability limitations in existing MANET routing protocols, including:
1. **Base Paper**: *EER-MANET-EFIAGNN* (Maya D. et al., *Knowledge-Based Systems*, Elsevier, 2025)
2. **Senior's Extension**: *AT-AEES-MANET* (Chilton J.)

---

### Architecture & Phases
- **Phase 1: Federated Trust Learning (FTL)**: Distributed trust estimation across 10 Cluster Heads without central aggregation, protected by $(\epsilon=0.1, \delta=10^{-5})$-Differential Privacy and Reputation-Weighted FedAvg.
- **Phase 2: Generative Adversarial Trust Model (GATM)**: Online adversarial generator synthesizing deceptive and zero-day attack vectors to continuously harden the discriminator.
- **Phase 3: Dynamic Bayesian Trust Calibration (BTC)**: Models trust as a $\text{Beta}(\alpha, \beta)$ posterior distribution with uncertainty bounds and graceful poisoning recovery.
- **Phase 4: Model-Agnostic Meta-Learning (MAML)**: Meta-trained across diverse attack scenarios for few-shot adaptation to novel attack dynamics within 5–10 interaction windows.

---

### Standalone Directory Layout
```
RESILIENT_MANET_STANDALONE/
├── config.py                              # Global network, energy & attack configurations
├── main_resilient.py                      # Master standalone runner for 10 Monte Carlo runs
├── modules/
│   ├── phase1_ftl.py                      # Phase 1: Distributed FTL with Differential Privacy
│   ├── phase2_gatm.py                     # Phase 2: Generative Adversarial Trust Model
│   ├── phase3_bayesian.py                 # Phase 3: Dynamic Bayesian Trust & Recovery
│   ├── phase4_maml.py                     # Phase 4: MAML Few-Shot Meta-Learner
│   └── resilient_manet_engine.py          # Unified Simulation & Routing Engine
└── outputs/
    ├── resilient_manet_execution.log      # Complete 10-run execution log
    └── resilient_manet_final_benchmark.json # Empirical benchmark results
```

---

### Execution
To execute the standalone simulation from scratch:
```bash
python3 RESILIENT_MANET_STANDALONE/main_resilient.py
```
