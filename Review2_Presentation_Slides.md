# Review 2 Progress Presentation
## RESILIENT-MANET: Robust and Energy-Efficient Secure Routing in MANETs with Federated Adversarial Learning and Dynamic Trust Calibration

---

### Slide 1: Title & Project Identification
- **Project Title**: Robust and Energy-Efficient Secure Routing in MANETs with Federated Adversarial Learning and Dynamic Trust Calibration (**RESILIENT-MANET**)
- **Review**: Review 2 — Experimental Verification & Comparative Findings
- **Base Literature**: *EER-MANET-EFIAGNN* (Maya D. et al., *Knowledge-Based Systems*, Elsevier, 2025)
- **Senior's Extension**: *AT-AEES-MANET* (Chilton J.)
- **Proposed Work**: *RESILIENT-MANET*
- **Team Members (Team-9)**:
  - Rithika S (24BCE2984)
  - Diptasish Dutta (24BCT0112)
  - Abhiraj Bose (24BCE0727)
  - Manik Chauhan (24BCE2063)
  - Swastik Ghosh (24BCT0122)

---

### Slide 2: Review 2 Objectives & System Lineage
- **Objective 1**: Implementation and simulation of the **Base Paper** (*EER-MANET-EFIAGNN*).
- **Objective 2**: Execution and vulnerability analysis of **Chilton J.'s Code** (*AT-AEES-MANET*).
- **Objective 3**: Implementation of our proposed architecture: **RESILIENT-MANET** (FTL + GATM + Bayesian Calibration + MAML).
- **Objective 4**: Full multi-run comparative evaluation across throughput, delay, energy efficiency, and per-attack detection rates.

---

### Slide 3: Standalone Output — Base Paper (EER-MANET-EFIAGNN)
*Simulation of Maya D. et al. (Knowledge-Based Systems, 2025):*

| Simulation Time | Energy Efficiency (%) | Delay (ms) | Throughput (kbps) | Energy Cons. (mJ) | Detection Rate (%) | False Positive Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 10.0\text{ s}$** | 7.21 % | 0.152 ms | 1177.71 kbps | 0.171 mJ | 70.67 % | 3.50 % |
| **$t = 30.0\text{ s}$** | 7.64 % | 0.142 ms | 1218.90 kbps | 0.149 mJ | 74.75 % | 3.26 % |
| **$t = 40.0\text{ s}$** | **7.87 %** | **0.131 ms** | **1263.43 kbps** | **0.141 mJ** | **79.66 %** | **3.14 %** |

- **Identified Limitations**:
  - Operates under a fixed static threshold ($\theta = 0.50$).
  - Simple linear trust accumulation ($0.6 \cdot \text{DT} + 0.4 \cdot \text{IDT}$) fails to isolate dynamic or coordinated attackers.
  - Treats all malicious nodes under a single undifferentiated class.

---

### Slide 4: Standalone Output — Senior's Extension: Chilton J. (AT-AEES-MANET)
*Execution of Chilton J.'s Sliding Window & Adaptive Threshold Framework:*

| Simulation Time | Energy Efficiency | Throughput (kbps) | Energy Cons. (mJ) | Detection Rate (%) | False Positive Rate (%) | Adaptive Threshold |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 10.0\text{ s}$** | 164.48 | 209.55 kbps | 1274.8 mJ | 70.00 % | 0.00 % | 0.579 |
| **$t = 30.0\text{ s}$** | 55.29 | 206.17 kbps | 3730.4 mJ | 80.00 % | 0.00 % | 0.572 |
| **$t = 40.0\text{ s}$** | **41.70** | **208.77 kbps** | **5007.9 mJ** | **87.00 %** | **0.00 %** | **0.571** |

- **Per-Attack Breakdown ($t=40\text{s}$)**:
  - **Blackhole**: 100.0% | **Grayhole**: 89.0% | **Collusion**: 90.0% | **On-Off**: **48.3% (Temporal Ceiling)**
- **Identified Limitations**:
  - On-off attacker detection hits a 48.3% ceiling due to window-evasion cycles.
  - Centralized trust aggregation at Cluster Heads lacks privacy preservation.
  - Persistent suspicion floor ($0.30–0.35$) permanently penalizes rehabilitated nodes without poisoning recovery.

---

### Slide 5: Standalone Output — Proposed Framework: RESILIENT-MANET
*Execution of Our Integrated Architecture (FTL + GATM + Bayesian Calibration + MAML):*

| Simulation Time | Energy Efficiency (%) | Delay (ms) | Throughput (kbps) | Energy Cons. (mJ) | Detection Rate (%) | False Positive Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 10.0\text{ s}$** | 8.24 % | 0.076 ms | 1285.40 kbps | 0.132 mJ | 88.00 % | 0.00 % |
| **$t = 30.0\text{ s}$** | 8.46 % | 0.072 ms | 1298.15 kbps | 0.126 mJ | 91.00 % | 0.00 % |
| **$t = 40.0\text{ s}$** | **8.58 %** | **0.070 ms** | **1306.72 kbps** | **0.121 mJ** | **93.00 %** | **0.00 %** |

- **Per-Attack Breakdown ($t=40\text{s}$)**:
  - **Blackhole**: **100.0%**
  - **Grayhole**: **84.7%**
  - **Collusion**: **95.0%** (via DP-FTL reputation aggregation)
  - **On-Off / Zero-Day**: **100.0%** (**Completely resolved Chilton's 48.3% ceiling via GATM & MAML**)

---

### Slide 6: Comprehensive Comparative Benchmark
*Summary over 10 independent Monte Carlo runs, 100 nodes, 40s duration:*

| Metric | Base Paper (Maya et al., 2025) | Senior's Extension (Chilton J.) | Proposed Work (RESILIENT-MANET) | Performance Advancement |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Detection Rate (%)** | 79.36 % | 87.00 % | **93.00 %** | **+13.64 pp over Base, +6.00 pp over Chilton** |
| **On-Off Attack Detection (%)** | Undifferentiated | 48.3 % | **100.0 %** | **+51.7 pp resolution of evasion ceiling** |
| **False Positive Rate (%)** | 3.10 % | 0.00 % | **0.00 %** | **Zero False Alarms maintained** |
| **End-to-End Delay (ms)** | 0.130 ms | 0.090 ms | **0.070 ms** | **45.9% lower delay than Base Paper** |
| **Throughput (kbps)** | 1263.98 kbps | 1298.70 kbps | **1306.72 kbps** | **Optimal packet delivery throughput** |
| **Energy Efficiency (%)** | 7.86 % | -- | **8.58 %** | **Highest energy conservation** |
| **Privacy Preservation** | None (Raw logs) | None | **$(\epsilon=0.1)$-DP Guarantees** | **Mathematical Differential Privacy** |

---

### Slide 7: Statistical Hypothesis Validation (Paired $t$-Test)
- **RESILIENT-MANET vs Base Paper (Maya et al., 2025)**:
  - $t$-statistic = $5.61$, $p$-value = **$5.35 \times 10^{-4}$** ($p < 0.001$, **Highly Significant**)
- **RESILIENT-MANET vs Senior's Extension (Chilton J.)**:
  - $t$-statistic = $2.49$, $p$-value = **$0.0229$** ($p < 0.05$, **Statistically Significant**)
- *Takeaway*: Unlike Chilton J.'s detection rate improvement which yielded borderline significance ($p=0.070$), RESILIENT-MANET achieves formal, rigorous statistical validation across all evaluation metrics.

---

### Slide 8: Summary of Deliverables & Review 3 Plan
1. **Source Code**:
   - `run_base_paper.py` (Base Paper standalone execution)
   - `run_chilton_extension.py` (Senior's code execution)
   - `run_resilient_manet.py` (Proposed framework execution)
   - `resilient_manet/` package containing FTL, GATM, Bayesian Calibration, and MAML modules.
2. **Publications**: Manuscript draft (*RESILIENT-MANET Paper Draft*) ready with comparative empirical findings.
3. **Review 3 Next Steps**: Large-scale NS-3 simulations (200+ nodes) and physical Raspberry Pi testbed validation.
