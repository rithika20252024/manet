#!/usr/bin/env python3
"""
run_all_review2.py
===================
Master execution script for Review 2:
1. Executes Base Paper (A), Senior Extension (B), and Proposed RESILIENT-MANET (C)
2. Runs 10 Monte Carlo simulation runs across 100 mobile nodes
3. Generates 3-way comparative tables & JSON summaries
4. Outputs statistical significance (paired t-tests)
5. Prints quick summary ready for Review 2 presentation
"""

import sys
import os

# Set path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resilient_manet.evaluation.benchmark_abc import run_benchmark

def main():
    print("\n" + "#"*80)
    print("MASTER WORKFLOW: RESILIENT-MANET")
    print("#"*80 + "\n")
    
    # Run full 10-run 
    summary = run_benchmark(n_runs=10, n_nodes=100, sim_time=40.0, seed=42)
    
    print("\n" + "#"*80)
    
    print("  1. Codebase : /Users/rithika/Documents/CODE 2/resilient_manet/")
    print("  2. Results: outputs/review2_benchmark_abc.json")
   print("#"*80 + "\n")

if __name__ == '__main__':
    main()
