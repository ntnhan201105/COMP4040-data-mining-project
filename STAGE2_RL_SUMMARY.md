# Stage 2: RL Experiments — Summary Report
## 1. Introduction
This report presents the results of **Stage 2: Reinforcement Learning Experiments**for EV charging scheduling. We train four RL algorithms (DQN, PPO, DDPG, SAC)and compare them against heuristic baselines and the MPC Oracle upper bound.
**Narrative: "From Heuristics to Intelligence"**
---
## 2. Learning Curves
![Learning Curves](stage2/output/caltech_rl_learning_curves.png)
### Training Summary
| Agent | Episodes | Time (s) | Final Satisfaction |
|-------|----------|----------|-------------------|
| DQN_base | 104 | 58.8 | 0.981 |
| PPO_base | 105 | 109.2 | 0.656 |
| DDPG_base | 104 | 190.0 | 0.680 |
| SAC_base | 104 | 304.4 | 0.698 |
| PPO_weather | 105 | 107.9 | 0.687 |
| SAC_weather | 104 | 310.0 | 0.666 |

---
## 3. Performance Comparison
![Performance](stage2/output/caltech_rl_performance_comparison.png)

### Detailed Metrics
| policy       |   satisfaction_ratio |   peak_demand_kw |   jain_fairness |
|:-------------|---------------------:|-----------------:|----------------:|
| DDPG         |                0.763 |           51.176 |           0.816 |
| DQN          |                0.971 |           70.126 |           0.995 |
| EDF          |                0.974 |           62.251 |           0.995 |
| FCFS         |                0.974 |           62.197 |           0.993 |
| MPC Oracle   |                0.974 |           21.099 |           0.992 |
| PPO          |                0.464 |           26.22  |           0.557 |
| PPO +W       |                0.492 |           31.392 |           0.601 |
| Round-Robin  |                0.972 |           59.339 |           0.995 |
| SAC          |                0.781 |           53.812 |           0.874 |
| SAC +W       |                0.763 |           55.484 |           0.886 |
| Uncontrolled |                0.974 |           67.389 |           0.995 |

---
## 4. RL Power Profiles
![Power Profiles](stage2/output/caltech_rl_power_profiles.png)

---
## 5. Radar Chart
![Radar](stage2/output/caltech_rl_radar.png)

---
## 6. Weather Ablation
![Weather Ablation](stage2/output/caltech_rl_weather_ablation.png)

|                |   satisfaction_ratio |
|:---------------|---------------------:|
| ('PPO', False) |             0.463667 |
| ('PPO', True)  |             0.491882 |
| ('SAC', False) |             0.781225 |
| ('SAC', True)  |             0.763438 |

**Interpretation:** Comparing weather-aware vs weather-blind agents across different weather conditions reveals whether weather context helps RL agents make better scheduling decisions.

---
## 7. DQN Strategy Selection
![DQN Timeline](stage2/output/caltech_rl_dqn_timeline.png)

### Strategy Usage Frequency
| strategy     |   count |
|:-------------|--------:|
| Uncontrolled |     440 |
| Conservative |     294 |
| Round-Robin  |     277 |
| EDF          |     217 |
| FCFS         |     212 |

**Interpretation:** The DQN meta-learner reveals which heuristic is most appropriate at different times of day and demand levels.

---
## 8. Key Findings
1. **RL vs Heuristics**: …
2. **RL vs MPC Oracle**: …
3. **Weather Impact**: …
4. **Algorithm Comparison**: …

*(Fill in findings after reviewing the results above.)*
