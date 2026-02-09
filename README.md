# Frozen Lake Reinforcement Learning (DQN-Style)

This repository contains professional-grade implementations of **Q-Learning** and **SARSA** algorithms applied to the `FrozenLake-v1` environment from OpenAI Gymnasium. The codebase is structured to match the architectural style of high-level Deep Q-Network (DQN) scripts, featuring modular training/deployment methods and persistent data storage.

## Project Overview

The goal is to navigate a 4x4 grid (Frozen Lake) from the Start (S) to the Goal (G) while avoiding Holes (H). The environment is set to `is_slippery=True`, adding a stochastic layer to the reinforcement learning challenge.

### Core Features
- **Modular Architecture:** Separate `train()` and `deploy()` functions for high-speed computation vs. human-mode visualization.
- **CPU-Optimized Design:** As tabular methods (Q-Learning/SARSA) rely on NumPy for matrix operations, the scripts are hardcoded for **CPU execution** to avoid unnecessary hardware-check overhead while maintaining a research-ready structural template.
- **Persistent Storage:** 
  - `.npy` files store the trained Q-Tables (The Agent's "Brain").
  - `.npz` files store the training rewards history for performance analysis.
- **Visualization:** Automatic generation of smoothed success rate curves and Q-Table heatmaps in the `/results` directory.

---

## Performance Comparison

Comparing the success rate (moving average) shows how each algorithm converges.

| Q-Learning Success Rate | SARSA Success Rate |
| :---: | :---: |
| ![Q-Learning Reward](results/q_learning_results.png) | ![SARSA Reward](results/sarsa_success_rate.png) |

---

## Q-Table Visualization

The heatmaps visualize the learned Q-values for each of the 16 states across the 4 possible actions (Left, Down, Right, Up). 

| Q-Learning Heatmap | SARSA Heatmap |
| :---: | :---: |
| ![Q-Table QL](results/q_table_heatmap_q_learning.png) | ![SARSA-Table](results/sarsa_q_table_heatmap.png) |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Gymnasium
- NumPy, Matplotlib, Seaborn

### Running the Scripts
To train a new model or deploy a trained one, toggle the `MODE` variable in the `main()` function:
- `MODE = 'new'`: Train from scratch.
- `MODE = 'resume'`: Load existing data and continue training.
- `MODE = 'deploy'`: Watch the agent play in "Human" render mode.

```bash
python Q-learning-Frozen-Lake.py
python SARSA-Frozen-Lake.py