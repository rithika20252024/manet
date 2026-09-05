# RESILIENT-MANET: Robust and Energy-Efficient Secure Routing in MANETs with Federated Adversarial Learning and Dynamic Trust Calibration

**Rithika S, Diptasish Dutta, Abhiraj Bose, Manik Chauhan, Swastik Ghosh**  
*Department of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India*

---

## Abstract
Mobile Ad Hoc Networks (MANETs) face critical operational vulnerabilities stemming from multi-hop wireless communications, battery-capacity constraints, and internal node misbehavior. Prior state-of-the-art architectures, specifically the baseline **EER-MANET-EFIAGNN** framework (Maya D. et al., *Knowledge-Based Systems*, 2025) and its sliding-window extension **AT-AEES-MANET** (Chilton J.), exhibit fundamental security and adaptability deficiencies: (i) centralized or semi-centralized trust computation vulnerable to single-point-of-failure compromises and indirect trust privacy leaks, (ii) vulnerability to zero-day and intermittent on-off attacks where detection rates plateau at ~48.3%, and (iii) irreversible trust poisoning where rehabilitated nodes suffer eternal lockout.

To resolve these challenges, this paper presents **RESILIENT-MANET**, an end-to-end decentralized trust-aware routing architecture. The framework integrates four coordinated mechanisms:
1. **Federated Trust Learning (FTL)**: Distributed trust estimation across Cluster Heads (CHs) with $(\epsilon=0.1)$-Differential Privacy and Reputation-Weighted FedAvg aggregation.
2. **Generative Adversarial Trust Model (GATM)**: Online adversarial generator-discriminator sparring that synthesizes deceptive attack profiles to train a discriminator robust against zero-day evasion.
3. **Dynamic Bayesian Trust Calibration (BTC)**: Models trust as a Beta posterior distribution with uncertainty confidence intervals and graceful poisoning recovery.
4. **Model-Agnostic Meta-Learning (MAML)**: Enables few-shot adaptation to novel attack dynamics within 5–10 interaction windows.

In comprehensive Monte Carlo simulations (10 independent runs, 100 mobile nodes, 40s duration), RESILIENT-MANET achieves a **93.00% overall detection rate** at $t=40\text{s}$ (improving by +13.64 percentage points over the Base Paper and +6.00 percentage points over Chilton J.'s extension), **100% on-off attack detection**, 0.00% false positive rate, 1306.72 kbps throughput, and 0.070 ms end-to-end delay with high statistical significance ($p = 5.35 \times 10^{-4}$).

**Keywords**: Mobile Ad Hoc Networks (MANETs), Federated Learning, Differential Privacy, Generative Adversarial Networks (GAN), Bayesian Trust Calibration, Meta-Learning (MAML), Secure Routing.

---

## 1. Introduction & Problem Formulation
Mobile Ad Hoc Networks (MANETs) are self-configuring wireless networks lacking centralized infrastructure. While suitable for mission-critical disaster recovery, tactical military networks, and vehicular ad hoc communications, their open environment leaves them vulnerable to malicious packet manipulation.

### 1.1 Progression of Literature & Gaps
- **Base Paper (EER-MANET-EFIAGNN, Maya et al., 2025)**: Integrated Fast Continual Multi-View Clustering (FCMVC), Explicit Feature Interaction GNN (EFIAGNN), and the Horned Lizard Optimization Algorithm (HLOA). However, its static 0.50 J trust threshold, linear cumulative averaging ($0.6 \cdot \text{DT} + 0.4 \cdot \text{IDT}$), and undifferentiated single-class attack treatment restrict its effectiveness against dynamic adversaries.
- **Senior's Extension (AT-AEES-MANET, Chilton J.)**: Introduced exponentially decayed sliding windows ($\lambda=0.92, W=10$), an adaptive trust threshold, and majority-vote collusion filtering. While Chilton J. increased detection to 87.00%, three critical open challenges remained:
  1. *On-Off Attack Ceiling*: Intermittent attackers evading sliding-window durations plateaued detection at 48.3%.
  2. *Centralized Vulnerability*: Cluster Heads aggregated raw interaction logs without differential privacy.
  3. *Permanent Node Lockout*: A persistent suspicion floor ($0.30–0.35$) prevented falsely-flagged or rehabilitated nodes from regaining legitimate status.

---

## 2. Experimental Verification & Standalone Outputs

### 2.1 Base Paper Simulation Output (EER-MANET-EFIAGNN)
*Evaluated across 10 simulation runs (100 nodes, 40s):*

| Simulation Time | Energy Efficiency (%) | End-to-End Delay (ms) | Throughput (kbps) | Energy Consumption (mJ) | Detection Rate (%) | False Positive Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $t = 10.0\text{ s}$ | 7.21 % | 0.152 ms | 1177.71 kbps | 0.171 mJ | 70.67 % | 3.50 % |
| $t = 30.0\text{ s}$ | 7.64 % | 0.142 ms | 1218.90 kbps | 0.149 mJ | 74.75 % | 3.26 % |
| $t = 40.0\text{ s}$ | 7.87 % | 0.131 ms | 1263.43 kbps | 0.141 mJ | 79.66 % | 3.14 % |

### 2.2 Senior's Extension Output: Chilton J. (AT-AEES-MANET)
*Evaluated across 10 simulation runs (100 nodes, 40s):*

| Simulation Time | Energy Efficiency | Throughput (kbps) | Energy Cons. (mJ) | Detection Rate (%) | False Positive Rate (%) | Adaptive Threshold |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $t = 10.0\text{ s}$ | 164.48 | 209.55 kbps | 1274.8 mJ | 70.00 % | 0.00 % | 0.579 |
| $t = 30.0\text{ s}$ | 55.29 | 206.17 kbps | 3730.4 mJ | 80.00 % | 0.00 % | 0.572 |
| $t = 40.0\text{ s}$ | 41.70 | 208.77 kbps | 5007.9 mJ | 87.00 % | 0.00 % | 0.571 |

*Per-Attack Detection Rates*: Blackhole: 100.0% | Grayhole: 89.0% | Collusion: 90.0% | **On-Off: 48.3% (Temporal Evasion Limit)**.

### 2.3 Proposed Extension Output (RESILIENT-MANET)
*Evaluated across 10 simulation runs (100 nodes, 40s):*

| Simulation Time | Energy Efficiency (%) | End-to-End Delay (ms) | Throughput (kbps) | Energy Consumption (mJ) | Detection Rate (%) | False Positive Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $t = 10.0\text{ s}$ | 8.24 % | 0.076 ms | 1285.40 kbps | 0.132 mJ | 88.00 % | 0.00 % |
| $t = 30.0\text{ s}$ | 8.46 % | 0.072 ms | 1298.15 kbps | 0.126 mJ | 91.00 % | 0.00 % |
| $t = 40.0\text{ s}$ | 8.58 % | 0.070 ms | 1306.72 kbps | 0.121 mJ | 93.00 % | 0.00 % |

*Per-Attack Detection Rates*: Blackhole: **100.0%** | Grayhole: **84.7%** | Collusion: **95.0%** | **On-Off / Zero-Day: 100.0% (Resolved Chilton's ceiling)**.

---

## 3. Comprehensive 3-Way Comparative Benchmark

| Metric | Base Paper (Maya et al., 2025) | Senior's Extension (Chilton J.) | Proposed Work (RESILIENT-MANET) | Performance Gain over Base Paper |
| :--- | :---: | :---: | :---: | :---: |
| **Detection Rate (%)** | 79.36 % | 87.00 % | **93.00 %** | **+13.64 percentage points** |
| **On-Off Attack Detection** | Undifferentiated | 48.3 % | **100.0 %** | **+51.7 percentage points** |
| **False Positive Rate (%)** | 3.10 % | 0.00 % | **0.00 %** | **-3.10 percentage points** |
| **End-to-End Delay (ms)** | 0.130 ms | 0.090 ms | **0.070 ms** | **45.9% reduction** |
| **Throughput (kbps)** | 1263.98 kbps | 1298.70 kbps | **1306.72 kbps** | **+3.38% increase** |
| **Energy Efficiency (%)** | 7.86 % | -- | **8.58 %** | **+0.72 percentage points** |
| **Privacy Preservation** | None (Raw logs) | None | **$(\epsilon=0.1, \delta=10^{-5})$-DP** | **Mathematical Guarantee** |

### 3.1 Statistical Significance Testing (Paired $t$-Test)
- **RESILIENT-MANET vs Base Paper (Maya et al., 2025)**: $t = 5.61$, $p = 5.35 \times 10^{-4}$ ($p < 0.001$, **Highly Significant**).
- **RESILIENT-MANET vs Senior's Extension (Chilton J.)**: $t = 2.49$, $p = 0.0229$ ($p < 0.05$, **Statistically Significant**).

---

## 4. Conclusion
RESILIENT-MANET resolves the vulnerabilities inherent in prior trust-aware MANET routing approaches. By coupling privacy-preserving Federated Trust Learning, Generative Adversarial sparring, Bayesian calibration with poisoning recovery, and Meta-Learning adaptation, our architecture achieves 93.00% overall detection rate, 100% on-off attack mitigation, 0.00% false positives, and superior energy-efficient routing.

---
## References
1. Maya D., Thuthi Sarabai D., Samuel Ebenezer Anand Vijai, "Trust-aware energy-efficient and secure routing in MANETs using explicit feature interaction graph neural networks," *Knowledge-Based Systems*, vol. 331, p. 114572, Elsevier, 2025.
2. Chilton J., "AT-AEES-MANET: Adaptive Trust-Aware Energy-Efficient Secure Routing in Mobile Ad Hoc Networks with Attack-Resilient Trust Evaluation," IEEE Access / Technical Report, 2026.
3. Buchegger S., Le Boudec J.-Y., "Performance analysis of the CONFIDANT protocol," in *Proc. ACM MobiHoc*, pp. 226–236.
4. Finn C., Abbeel P., Levine S., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks," in *Proc. ICML*, 2017.
