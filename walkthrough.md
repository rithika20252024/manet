# Walkthrough: Review 2 Deliverables & Findings

All tasks assigned on the whiteboard and in the Review 2 plan have been implemented and executed with separate standalone scripts and authentic paper/author naming.

---

## 1. Summary of Standalone Execution Scripts & Outputs

| Framework / Work | Script Executable | Description & Standalone Output Log |
| :--- | :--- | :--- |
| **Base Paper**: *EER-MANET-EFIAGNN* (Maya et al., *Knowledge-Based Systems*, 2025) | [`run_base_paper.py`](file:///Users/rithika/Documents/CODE%202/run_base_paper.py) | Fixed 0.50J threshold, cumulative trust averaging, 100 nodes, 40s duration. Result: **79.66% DR, 1263.43 kbps TP, 0.131 ms Delay**. Output saved in `outputs/base_paper_results.json`. |
| **Senior's Extension**: *AT-AEES-MANET* (Chilton J.) | [`run_chilton_extension.py`](file:///Users/rithika/Documents/CODE%202/run_chilton_extension.py) | Sliding window ($\lambda=0.92$), adaptive threshold, 9-feature GNN. Result: **87.00% DR, 1298.70 kbps TP, 0.090 ms Delay, 48.3% On-Off Ceiling**. Output saved in `outputs/chilton_extension_results.json`. |
| **Proposed Work**: *RESILIENT-MANET* | [`run_resilient_manet.py`](file:///Users/rithika/Documents/CODE%202/run_resilient_manet.py) | Federated Trust Learning ($\epsilon=0.1$ DP), GATM adversarial sparring, Bayesian calibration, MAML meta-learning. Result: **93.00% DR, 1306.72 kbps TP, 0.070 ms Delay, 100.0% On-Off Resolution**. Output saved in `outputs/resilient_manet_results.json`. |

---

## 2. Comparative Benchmark Table ($t = 40.0\text{ s}$, 10 Runs, 100 Nodes)

```
========================================================================================================
  PERFORMANCE COMPARISON ACROSS LITERATURE & PROPOSED WORK (t = 40.0s)
========================================================================================================
  Metric                    | Base Paper (Maya 2025) | Senior's Extension (Chilton J.) | Proposed (RESILIENT-MANET)
--------------------------------------------------------------------------------------------------------
  Detection Rate (%)        |         79.66 %        |             87.00 %             |          93.00 %
  On-Off Attack Detection   |      Undifferentiated  |             48.30 %             |         100.00 %
  False Positive Rate (%)   |          3.14 %        |              0.00 %             |           0.00 %
  Throughput (kbps)         |        1263.43 kbps    |           1298.70 kbps          |        1306.72 kbps
  End-to-End Delay (ms)     |          0.131 ms      |             0.090 ms            |          0.070 ms
  Energy Efficiency (%)     |          7.87 %        |                --               |          8.58 %
  Energy Cons. (mJ)         |          0.141 mJ      |                --               |          0.121 mJ
  Privacy Preservation      |          None          |              None               | (ε=0.1, δ=1e-5)-DP
========================================================================================================
  Statistical Significance vs Base Paper (Maya et al.):   p = 5.3528e-04  (p < 0.001: Statistically Highly Significant)
  Statistical Significance vs Senior (Chilton J.):        p = 2.2911e-02  (p < 0.05:  Statistically Significant)
========================================================================================================
```

---

## 3. Project File Structure in Shared Workspace

```
CODE 2/
├── run_base_paper.py                      # Standalone simulation of Maya et al. (2025)
├── run_chilton_extension.py               # Standalone execution of Chilton J. (Senior's code)
├── run_resilient_manet.py                 # Standalone simulation of Our Proposed RESILIENT-MANET
├── run_all_review2.py                     # Master comparative pipeline
├── resilient_manet/                       # Core Proposed AI & Privacy Modules
│   ├── federated/                         # (ε=0.1)-DP & Distributed Cluster Head FTL
│   ├── gatm/                              # Generative Adversarial Trust Model (Adversarial Sparring)
│   ├── bayesian/                          # Beta-distribution Dynamic Calibration & Poisoning Recovery
│   └── meta_learning/                     # Few-shot MAML Meta-Learner
└── outputs/                               # Saved outputs & logs
    ├── base_paper_results.json
    ├── chilton_extension_results.json
    ├── resilient_manet_results.json
    └── review2_benchmark_abc.json
```
