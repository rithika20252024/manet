# Node Configuration & Simulation Parameters
## RESILIENT-MANET: Robust and Energy-Efficient Secure Routing in MANETs with Federated Adversarial Learning and Dynamic Trust Calibration

---

## 1. Physical & Hardware Node Specifications

| Parameter | Value / Setting | Description |
| :--- | :--- | :--- |
| **Total Number of Nodes ($N$)** | **$100\text{ mobile nodes}$** | Standard simulation scale (scalable to $200+$ nodes) |
| **Initial Node Battery Energy ($E_0$)** | **$15.1\text{ Joules}$** | Initial energy budget per mobile node |
| **Radio Transmission Range ($R_{\text{tx}}$)** | **$250.0\text{ metres}$** | Maximum omnidirectional wireless broadcast reach |
| **Antenna Model** | **Omnidirectional** | Uniform $360^\circ$ signal coverage |
| **Physical Interface Type** | **WirelessPhy (IEEE 802.11b)** | Direct-Sequence Spread Spectrum (DSSS) |
| **Nominal Channel Bandwidth** | **$2\text{ Mbps}$** | 802.11b standard wireless channel rate |
| **Memory Footprint** | **$< 128\text{ MB}$ per node** | Lightweight memory-bound execution for edge/Raspberry Pi deployment |

---

## 2. Terrain & Mobility Model Configuration

| Mobility Parameter | Value / Specification | Details |
| :--- | :--- | :--- |
| **Simulation Terrain Area** | **$1000\,\text{m} \times 1000\,\text{m}$ ($1\,\text{km}^2$)** | 2D continuous simulation area without boundaries |
| **Mobility Model** | **Random Waypoint (RWP)** | Dynamic trajectory generation |
| **Node Speed Range ($v$)** | **$1.0\text{ m/s} - 10.0\text{ m/s}$** | Uniform random distribution ($3.6\text{ km/h} - 36\text{ km/h}$) |
| **Pause Time ($t_{\text{pause}}$)** | **$5.0\text{ seconds}$** | Pause duration upon reaching destination waypoint |
| **Propagation Model** | **Two-Ray Ground Model** | Accurate for ground reflection in ad hoc scenarios |

---

## 3. Traffic & Energy Dissipation Model

### 3.1 Network Traffic Generation
- **Traffic Agent Type**: Constant Bit Rate (**CBR**) over **UDP**
- **Data Packet Size**: **$512\text{ bytes}$** ($4096\text{ bits}$)
- **Packet Sending Rate**: **$4\text{ packets/second}$** per active node flow
- **Simulation Duration**: **$40.0\text{ seconds}$** (evaluated at snapshots $t=10\text{s}, 30\text{s}, 40\text{s}$)

### 3.2 Radio Energy Model (First-Order Radio Model)
- **Transmitter Electronics Energy ($E_{\text{tx}}$)**: $50\text{ nJ/bit}$ ($50 \times 10^{-9}\text{ J/bit}$)
- **Receiver Electronics Energy ($E_{\text{rx}}$)**: $50\text{ nJ/bit}$ ($50 \times 10^{-9}\text{ J/bit}$)
- **Power Amplifier Energy ($\epsilon_{\text{amp}}$)**: $100\text{ pJ/bit/m}^2$ ($100 \times 10^{-12}\text{ J/bit/m}^2$)
- **Energy for Transmitting a $k$-bit packet over distance $d$**:
  $$E_{\text{Tx}}(k, d) = k \cdot E_{\text{tx}} + k \cdot \epsilon_{\text{amp}} \cdot d^2$$
- **Energy for Receiving a $k$-bit packet**:
  $$E_{\text{Rx}}(k) = k \cdot E_{\text{rx}}$$

---

## 4. 9-Dimensional Node Feature Representation Vector ($h_i$)

Each mobile node $i$ maintains a normalized **9-dimensional feature vector** $h_i \in [0, 1]^9$ used by the GNN, FTL, and GATM models:

$$h_i = \left[ h_{i,1},\, h_{i,2},\, h_{i,3},\, h_{i,4},\, h_{i,5},\, h_{i,6},\, h_{i,7},\, h_{i,8},\, h_{i,9} \right]$$

| Index | Feature Symbol | Formulation / Range | Physical & Security Meaning |
| :---: | :--- | :--- | :--- |
| **1** | **Normalized $X$-Coord** | $x_i / \text{AreaWidth} \in [0, 1]$ | Spatial 2D horizontal position |
| **2** | **Normalized $Y$-Coord** | $y_i / \text{AreaHeight} \in [0, 1]$ | Spatial 2D vertical position |
| **3** | **Residual Energy Ratio** | $E_i(t) / E_0 \in [0, 1]$ | Remaining battery capacity ratio |
| **4** | **Bayesian Posterior Trust** | $\mu_i = \frac{\alpha_i}{\alpha_i + \beta_i} \in [0, 1]$ | Expected packet forwarding reliability |
| **5** | **Trust Stability Score ($\sigma_i$)** | $1.0 - \min\left(1.0, 2.0 \cdot \text{Std}(\Theta_i)\right)$ | Inverse of trust variance (consistency metric) |
| **6** | **Cluster Role Flag** | $1.0\text{ (Cluster Head)}, 0.5\text{ (Member)}$ | Hierarchical role in the FTL group |
| **7** | **Centroid Proximity ($q_i$)** | $1.0 - \frac{\|p_i - p_{\text{center}}\|}{\text{AreaWidth}}$ | Distance metric to network topological center |
| **8** | **Attack Suspicion Score ($\phi_i$)** | $1.0 - \mu_i \in [0, 1]$ | Composite indicator of malicious likelihood |
| **9** | **Normalized Mobility Speed** | $v_i / v_{\text{max}} \in [0, 1]$ | Current instantaneous velocity |

---

## 5. Adversarial Node Distribution & Behavioral Profiles

A fraction of $\rho = 0.10$ (**10 nodes out of 100**) are configured as malicious adversaries across 5 distinct attack patterns:

```
Total Nodes: 100
├── Honest Nodes (90%): 90 nodes
└── Malicious Nodes (10%): 10 nodes
    ├── Blackhole Attack (25% of malicious): ~2-3 nodes
    ├── Grayhole Attack (25% of malicious): ~2-3 nodes
    ├── On-Off Attack (25% of malicious): ~2-3 nodes
    ├── Collusion Attack (15% of malicious): ~1-2 nodes
    └── Zero-Day Stealth Attack (10% of malicious): ~1 node
```

| Attack Category | Probability | Behavioral Configuration & Packet Forwarding Logic |
| :--- | :---: | :--- |
| **Honest (Benign)** | **$90\%$ of total** | Forwards packets honestly; occasional natural wireless loss ($p_{\text{loss}} \le 2\%$) |
| **Blackhole** | **$25\%$ of malicious** | **$100\%$ unconditional packet drop** on all received forwarding requests |
| **Grayhole** | **$25\%$ of malicious** | **Selective drop**: drops $60\%$ of data packets while forwarding routing control packets |
| **On-Off Attack** | **$25\%$ of malicious** | **Periodic duty cycle ($20\text{s}$ period)**: behaves honestly for $10\text{s}$, drops $100\%$ for $10\text{s}$ |
| **Collusion Attack** | **$15\%$ of malicious** | Drops $75\%$ of legitimate traffic while providing inflated recommendations ($+0.95$) for colluders |
| **Zero-Day Stealth** | **$10\%$ of malicious** | Intermittent camouflage dropping ($65\%$ drop rate) during burst traffic periods |

---

## 6. Cluster & Cluster Head (CH) Configuration for FTL

- **Total Clusters ($K$)**: **$10\text{ clusters}$**
- **Average Nodes per Cluster**: $\approx 10\text{ nodes / cluster}$
- **Cluster Head Selection Criteria**: Highest combined Bayesian posterior trust ($\mu_i$), residual energy ($E_i/E_0$), and spatial centroid proximity.
- **Privacy Parameter at CHs**: $(\epsilon=0.1, \delta=10^{-5})$-Differential Privacy Gaussian perturbation applied before transmitting weights to the global model.
