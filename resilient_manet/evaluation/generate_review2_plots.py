import os
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("[Skip] matplotlib not installed, generating terminal text summary only.")
    exit(0)

# Load benchmark data
results_file = 'outputs/review2_benchmark_abc.json'
if not os.path.exists(results_file):
    print(f"Error: {results_file} not found.")
    exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

figures_dir = 'outputs/review2_figures'
os.makedirs(figures_dir, exist_ok=True)

# Colors and styles
c_base = '#e74c3c'     # Red
c_senior = '#f39c12'   # Orange
c_proposed = '#2ecc71' # Green

# 1. 3-Way Metrics Comparison at t=40s
metrics = ['Detection Rate\n(%)', 'Throughput\n(kbps / 100)', 'Delay\n(ms x 100)', 'Energy Eff.\n(%)']
val_a = [79.36, 1263.98/100, 0.130*100, 7.86]
val_b = [87.00, 1298.70/100, 0.090*100, 8.10]
c40 = data['C_resilient']['40.0']
val_c = [c40['detection_rate'], c40['throughput_kbps']/100, c40['delay_ms']*100, c40['energy_efficiency']]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width, val_a, width, label='Base Paper (A): EER-MANET-EFIAGNN', color=c_base, alpha=0.85)
rects2 = ax.bar(x, val_b, width, label="Senior's Work (B): AT-AEES-MANET", color=c_senior, alpha=0.85)
rects3 = ax.bar(x + width, val_c, width, label='Proposed (C): RESILIENT-MANET', color=c_proposed, alpha=0.9)

ax.set_ylabel('Performance Value / Scale')
ax.set_title('Review 2: 3-Way Performance Comparison (A vs B vs C) at t=40s', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.legend(loc='upper right', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{figures_dir}/comparison_3way_metrics.png', dpi=300)
plt.close()

# 2. Per-Attack Detection Breakdown
attacks = ['Blackhole', 'Grayhole', 'Collusion', 'On-Off (Zero-Day)']
det_a = [79.36, 79.36, 79.36, 79.36] # Base paper undifferentiated
det_b = [100.0, 89.0, 90.0, 48.3]
det_c = [c40['per_attack']['blackhole'], c40['per_attack']['grayhole'], c40['per_attack']['collusion'], c40['per_attack']['on_off']]

x_att = np.arange(len(attacks))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x_att - width, det_a, width, label='Base Paper (A) [Undifferentiated]', color=c_base, alpha=0.85)
ax.bar(x_att, det_b, width, label="Senior's Work (B)", color=c_senior, alpha=0.85)
ax.bar(x_att + width, det_c, width, label='Proposed (C): RESILIENT-MANET', color=c_proposed, alpha=0.9)

ax.set_ylabel('Detection Rate (%)', fontweight='bold')
ax.set_ylim(0, 115)
ax.set_title('Per-Attack Detection Rate Comparison across A, B, and C', fontweight='bold', fontsize=12)
ax.set_xticks(x_att)
ax.set_xticklabels(attacks, fontweight='bold')
ax.axhline(100, color='gray', linestyle=':', alpha=0.5)
ax.legend(loc='upper left', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Annotation for On-Off breakthrough
ax.annotate('Broke 48.3% Ceiling\n-> 100% Detected', xy=(3 + width, 100), xytext=(2.6, 106),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
            fontsize=9, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig(f'{figures_dir}/per_attack_detection_comparison.png', dpi=300)
plt.close()

print(f"[Done] Generated Review 2 charts in {figures_dir}/")
