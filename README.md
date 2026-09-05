# RESILIENT-MANET: Robust and Energy-Efficient Secure Routing in MANETs with Federated Adversarial Learning and Dynamic Trust Calibration

## Review 2 Research Implementation & Comparative Evaluation

### Authors (Team-9)
- **Rithika S** (24BCE2984)
- **Diptasish Dutta** (24BCT0112)
- **Abhiraj Bose** (24BCE0727)
- **Manik Chauhan** (24BCE2063)
- **Swastik Ghosh** (24BCT0122)

---

## 1. Overview
This repository contains the complete experimental framework and comparative analysis for **Review 2**:
1. **Base Paper**: *EER-MANET-EFIAGNN* (Maya D. et al., *Knowledge-Based Systems*, Elsevier, 2025)
2. **Senior's Extension**: *AT-AEES-MANET* (Chilton J.)
3. **Our Proposed Work**: *RESILIENT-MANET* (Federated Trust Learning with $(\epsilon=0.1)$-Differential Privacy, Generative Adversarial Trust Model (GATM), Dynamic Bayesian Trust Calibration with Poisoning Recovery, and Model-Agnostic Meta-Learning (MAML)).

---

## 2. Repository Structure
```
.
├── run_base_paper.py              # Standalone simulation of Maya et al. (2025)
├── run_chilton_extension.py       # Standalone execution of Chilton J. (Senior's work)
├── run_resilient_manet.py         # Standalone execution of Our Proposed RESILIENT-MANET
├── run_all_review2.py             # Master runner executing 10-run 3-way comparative benchmark
├── config.py                      # Global simulation, energy, and mobility parameters
├── main.py                        # Core simulation harness
├── resilient_manet/               # Proposed AI & Privacy Modules
│   ├── federated/                 # Distributed Cluster Head FTL & Differential Privacy
│   ├── gatm/                      # Generative Adversarial Sparring Engine
│   ├── bayesian/                  # Beta Posterior Calibration & Poisoning Recovery
│   ├── meta_learning/             # MAML Few-Shot Meta-Learner
│   └── evaluation/                # 3-way Benchmark Engine & Plotting
├── Review2_Presentation_Slides.md # Complete slide-by-slide PPT content for VTOP
├── RESILIENT_MANET_Paper_Draft.md # Complete IEEE/Elsevier draft manuscript
├── RESILIENT_MANET_Technical_Note.md # Detailed technical architecture & math note
├── implementation_plan.md         # Review 2 plan artifact
├── walkthrough.md                 # Walkthrough artifact
└── outputs/                       # Empirical JSON results and execution logs
    ├── base_paper_results.json
    ├── chilton_extension_results.json
    ├── resilient_manet_results.json
    └── review2_benchmark_abc.json
```

---

## 3. Quick Start & Execution

### Run Base Paper (Maya et al., 2025):
```bash
python3 run_base_paper.py
```

### Run Senior's Extension (Chilton J.):
```bash
python3 run_chilton_extension.py
```

### Run Proposed Framework (RESILIENT-MANET):
```bash
python3 run_resilient_manet.py
```

### Run Master 10-Run 3-Way Comparative Benchmark:
```bash
python3 run_all_review2.py
```

---

## 4. Key Comparative Results ($t = 40.0\text{ s}$, 10 Runs, 100 Mobile Nodes)

| Metric | Base Paper (Maya et al., 2025) | Senior's Extension (Chilton J.) | Proposed (RESILIENT-MANET) | Performance Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Detection Rate (%)** | 79.66 % | 87.00 % | **93.00 %** | **+13.64 pp over Base, +6.00 pp over Senior** |
| **On-Off Attack Detection** | Undifferentiated | 48.30 % | **100.00 %** | **Resolved Chilton's temporal evasion ceiling** |
| **False Positive Rate (%)** | 3.14 % | 0.00 % | **0.00 %** | **Zero False Alarms** |
| **Throughput (kbps)** | 1263.43 kbps | 1298.70 kbps | **1306.72 kbps** | **+3.38% Increase** |
| **End-to-End Delay (ms)** | 0.131 ms | 0.090 ms | **0.070 ms** | **45.9% Reduction** |
| **Energy Efficiency (%)** | 7.87 % | -- | **8.58 %** | **Superior Energy Preservation** |
| **Energy Cons. (mJ)** | 0.141 mJ | -- | **0.121 mJ** | **Optimal Battery Life** |
| **Privacy Preservation** | None | None | **$(\epsilon=0.1, \delta=10^{-5})$-DP** | **Mathematical Differential Privacy** |

---

## 5. Statistical Significance
- **RESILIENT-MANET vs Base Paper (Maya et al., 2025)**: $p = 5.35 \times 10^{-4}$ ($p < 0.001$, **Highly Significant**)
- **RESILIENT-MANET vs Senior's Extension (Chilton J.)**: $p = 0.0229$ ($p < 0.05$, **Statistically Significant**)
