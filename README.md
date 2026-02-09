# Frozen Lake Reinforcement Learning

This repository contains professional implementations of **Q-Learning** and **SARSA** for the OpenAI Gymnasium `FrozenLake-v1` environment. The code structure follows a DQN-style architecture with separate training and deployment methods.

## Algorithms
- **Q-Learning (Off-Policy):** Learns the optimal policy by looking at the maximum possible future reward.
- **SARSA (On-Policy):** Learns by following the current policy and observing the actual next action taken.

## Visualizing Results
After running the scripts, check the `results/` folder for:
1. **Success Rate Plots:** Smoothed moving averages of agent performance.
2. **Q-Table Heatmaps:** Visualization of learned values for each state-action pair.

### Q-Table Comparison
| Q-Learning Heatmap | SARSA Heatmap |
| :---: | :---: |
| ![Q-Table](results/q_table_heatmap_q_learning.png) | ![SARSA-Table](results/sarsa_q_table_heatmap.png) |