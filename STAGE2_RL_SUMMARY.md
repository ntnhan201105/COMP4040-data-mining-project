# Stage 2: RL Experiments — Summary Report

## 1. Introduction

This report presents the results of **Stage 2: Reinforcement Learning Experiments** for EV charging scheduling at the Caltech ACN site (54 stations, 5-minute scheduling intervals, 24-hour episodes).

We train four RL algorithms — **DQN, PPO, DDPG, SAC** — and compare them against heuristic baselines and the MPC Oracle upper bound, following a narrative we call **"From Heuristics to Intelligence."**

### Narrative Arc

| Chapter | Research Question | Method |
|---------|------------------|--------|
| **1. The Baseline Gap** | How far are heuristics from optimal? | Quantify gap between baselines and MPC Oracle |
| **2. Learning to Schedule** | Can model-free RL close that gap? | Train DQN, PPO, DDPG, SAC on base env |
| **3. Does Weather Help?** | Does weather context improve scheduling? | Re-train PPO, SAC with climate features |
| **4. What Did the Agents Learn?** | Do RL agents discover the same patterns as Stage 1? | Analyze learned policies, connect to clusters |

### Experimental Setup

- **Environment**: ACN-Sim wrapper (Gymnasium), 54 charging stations, 288 timesteps/episode (24h at 5-min resolution)
- **Observation**: Per-station features (occupancy, remaining demand, time-to-departure, laxity, prev rate, energy fraction, time sin/cos) — shape `(54, 8)` base, `(54, 13)` with weather
- **Action**: Continuous `[0, 1]^54` (PPO/SAC/DDPG) or Discrete meta-strategy (DQN)
- **Reward**: Energy delivery − peak demand penalty − unfairness penalty
- **Training**: 60,000 timesteps per agent (~208 episodes), seed=42
- **Evaluation**: 20 held-out test days (Sep 20 – Oct 9, 2018)

---

## 2. RL Algorithm Implementation Details

### 2.1 Problem Formulation as a Markov Decision Process (MDP)

The EV charging scheduling problem is formulated as an MDP where at each 5-minute timestep, the agent observes the state of all 54 charging stations and decides how much power to allocate to each:

- **State** $s_t$: A `(54, 8)` matrix (or `(54, 13)` with weather). Each row corresponds to one charging station with 8 features:
  1. `is_occupied` — whether an EV is plugged in (0 or 1)
  2. `remaining_demand` — fraction of requested energy not yet delivered
  3. `time_until_departure` — normalized time remaining before the EV leaves
  4. `laxity` — slack time (time remaining minus minimum charge time), normalized
  5. `prev_charging_rate` — charging rate applied in the previous timestep
  6. `energy_delivered_fraction` — ratio of energy delivered to energy requested
  7. `time_sin` — sinusoidal encoding of time-of-day
  8. `time_cos` — cosinusoidal encoding of time-of-day

- **Action** $a_t$: Depends on the algorithm (see below)
- **Transition**: The ACN-Sim physics engine advances by one 5-minute interval, projecting the action to the nearest allowable pilot signal and enforcing infrastructure constraints

### 2.2 Reward Function

The reward balances three objectives at each timestep:

$$R(s_t, a_t) = \alpha \cdot E_{\text{delivered}} - \beta \cdot \left(\frac{P_{\text{aggregate}}}{P_{\text{capacity}}}\right)^2 - \gamma \cdot \sigma(\text{satisfaction})$$

| Component | Weight | Description |
|-----------|--------|-------------|
| **Energy delivery** ($\alpha = 1.0$) | + | Total kWh delivered to all EVs this timestep |
| **Peak demand penalty** ($\beta = 0.5$) | − | Squared ratio of aggregate power to site capacity (150 kW), penalizing demand spikes |
| **Unfairness penalty** ($\gamma = 0.1$) | − | Standard deviation of per-EV satisfaction ratios across active EVs |

### 2.3 Algorithm Implementations

All agents use the **Stable-Baselines3** library with a shared `MlpPolicy` network architecture of **2 hidden layers × 256 units** for fair comparison. Training uses `DummyVecEnv` (single-environment vectorization) with `Monitor` wrapping for episode statistics.

#### DQN — Deep Q-Network (Meta-Strategy Selector)

**Key design decision**: The raw action space (continuous rates for 54 stations) is intractable for DQN, which requires discrete actions. We designed a **meta-strategy DQN** where the agent selects one of 5 built-in scheduling heuristics at each timestep:

| Action ID | Strategy | Description |
|:---------:|----------|-------------|
| 0 | Uncontrolled | Charge all occupied stations at maximum rate |
| 1 | FCFS | First-Come-First-Served priority ordering |
| 2 | EDF | Earliest-Deadline-First priority ordering |
| 3 | Round-Robin | Equal capacity sharing across all stations |
| 4 | Conservative | 50% of maximum rate for all occupied stations |

This transforms the problem from "how much to charge each station" to "which scheduling philosophy to apply right now" — the agent learns _when_ each heuristic is most appropriate.

The `DiscreteSchedulingEnv` wrapper flattens the 2D observation to a 1D vector (432-dim for base, 702-dim with weather) for DQN's MLP input.

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | $1 \times 10^{-3}$ |
| Replay buffer size | 50,000 transitions |
| Learning starts | 500 steps |
| Batch size | 64 |
| Discount ($\gamma$) | 0.99 |
| Exploration schedule | $\epsilon$: 1.0 → 0.05 over 30% of training |
| Target network update | Every 500 steps |

#### PPO — Proximal Policy Optimization

PPO is an **on-policy** actor-critic algorithm that optimizes a clipped surrogate objective to ensure stable policy updates. The agent directly outputs a continuous action vector $a \in [0, 1]^{54}$, where each element represents the normalized charging rate for one station.

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | $3 \times 10^{-4}$ |
| Rollout length (`n_steps`) | 288 (= 1 full episode per update) |
| Mini-batch size | 64 |
| Optimization epochs per rollout | 10 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Value function coefficient | 0.5 |
| GAE lambda ($\lambda$) | 0.95 |
| Discount ($\gamma$) | 0.99 |
| Max gradient norm | 0.5 |

**Design note**: Setting `n_steps=288` means PPO collects exactly one full episode (24h simulation) before each policy update. The entropy bonus (`ent_coef=0.01`) encourages exploration across the 54-dimensional action space.

#### SAC — Soft Actor-Critic

SAC is an **off-policy** algorithm that maximizes a maximum-entropy objective, balancing reward maximization with policy entropy. This makes SAC well-suited for exploration in high-dimensional continuous action spaces. The agent maintains twin Q-networks, a policy network, and a temperature parameter ($\alpha$) that automatically tunes the entropy–reward tradeoff.

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | $3 \times 10^{-4}$ |
| Replay buffer size | 50,000 transitions |
| Learning starts | 500 steps |
| Batch size | 256 |
| Soft update coefficient ($\tau$) | 0.005 |
| Discount ($\gamma$) | 0.99 |
| Training frequency | 1 gradient step per env step |
| Entropy tuning | Automatic |

#### DDPG — Deep Deterministic Policy Gradient

DDPG is an **off-policy** algorithm that learns a deterministic policy $\mu(s)$ augmented with exploration noise. It uses a target actor-critic pair for stable Q-value estimation via Polyak averaging.

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | $1 \times 10^{-3}$ |
| Replay buffer size | 50,000 transitions |
| Learning starts | 500 steps |
| Batch size | 256 |
| Soft update coefficient ($\tau$) | 0.005 |
| Discount ($\gamma$) | 0.99 |
| Training frequency | 1 gradient step per env step |
| Exploration noise | Ornstein-Uhlenbeck ($\sigma = 0.1$) |

**Design note**: The OU noise process generates temporally correlated exploration, which is better suited for physical control tasks than independent Gaussian noise. However, the 54-dimensional noise vector makes exploration challenging — the agent must discover cooperative charging strategies across all stations simultaneously.

### 2.4 Weather-Augmented Observations

For the weather-aware variants (PPO +W, SAC +W), we extend the observation with **5 climate features** broadcast to every station row:

| Feature | Source | Normalization |
|---------|--------|---------------|
| Temperature (°C) | LA weather stations (Burbank, Downtown, El Monte, Whiteman) | Z-score |
| Relative Humidity (%) | Same | Z-score |
| Wind Speed (m/s) | Same | Z-score |
| Visibility (km) | Same | Z-score |
| Precipitation (mm) | Same | Z-score |

These features are looked up by (year, month, day, hour) from the pre-processed `average_all.psv` climate dataset and broadcast as 5 additional columns, changing the observation shape from `(54, 8)` → `(54, 13)`.

### 2.5 Training Infrastructure

- **Vectorization**: `DummyVecEnv` with `Monitor` wrapper for episode logging
- **Callback**: Custom `EpisodeMetricsCallback` records per-episode satisfaction ratio, peak demand, Jain fairness, and energy delivered/requested
- **Checkpointing**: Model, training metrics CSV, and config JSON saved to `stage2/models/{algo}_{site}_{variant}/`
- **Reproducibility**: All agents initialized with `seed=42`; environment sampling uses NumPy seeding

---

## 3. Learning Curves — Chapter 2: "Learning to Schedule"

![Learning Curves](stage2/output/caltech_rl_learning_curves.png)

### Training Summary

| Agent | Episodes | Time | Final Satisfaction (last-10 avg) |
|-------|----------|------|----------------------------------|
| DQN (meta-strategy) | 208 | 3.8 min | 0.949 |
| PPO | 209 | 7.4 min | 0.721 |
| DDPG | 208 | 15.6 hrs | 0.678 |
| SAC | 208 | 25.3 min | 0.797 |
| PPO +Weather | 209 | 9.6 min | 0.748 |
| SAC +Weather | 208 | 36.1 min | 0.787 |

**Key observations from training:**
- **DQN converges fastest** and reaches the highest training satisfaction (~95%). Its discrete meta-strategy space (choose among 5 heuristics) is much simpler to optimize than the continuous 54-dimensional action space.
- **SAC** is the best-performing continuous-action agent, stabilizing around 80% satisfaction.
- **PPO** trains quickly but plateaus around 59–72% — the on-policy algorithm struggles with the high-dimensional continuous action space.
- **DDPG** is extremely slow (15.6 hours) due to deterministic policy gradients with exploration noise in a 54-dim action space, and achieves the lowest satisfaction among all RL agents.

---

## 4. Performance Comparison — All Policies

![Performance](stage2/output/caltech_rl_performance_comparison.png)

### Evaluation Results (mean over 20 test days)

| Policy | Satisfaction | Peak Demand (kW) | Jain Fairness |
|--------|:-----------:|:----------------:|:-------------:|
| **MPC Oracle** | **96.2%** | **53.7** | 0.980 |
| Uncontrolled | 96.2% | 144.9 | 0.998 |
| EDF | 96.0% | 113.7 | 0.998 |
| FCFS | 95.9% | 114.1 | 0.995 |
| Round-Robin | 95.5% | 105.4 | 0.997 |
| **DQN** | **95.5%** | **105.4** | **0.997** |
| SAC | 79.8% | 108.8 | 0.868 |
| SAC +W | 74.9% | 102.5 | 0.837 |
| DDPG | 71.9% | 100.4 | 0.765 |
| PPO +W | 63.8% | 71.9 | 0.673 |
| PPO | 59.3% | 65.5 | 0.618 |

---

## 5. RL Power Profiles

![Power Profiles](stage2/output/caltech_rl_power_profiles.png)

---

## 6. Radar Chart — Multi-Metric Comparison

![Radar](stage2/output/caltech_rl_radar.png)

---

## 7. Weather Ablation — Chapter 3: "Does Weather Help?"

![Weather Ablation](stage2/output/caltech_rl_weather_ablation.png)

### Weather-Aware vs Weather-Blind

| Agent | Without Weather | With Weather | Δ Satisfaction |
|-------|:--------------:|:------------:|:--------------:|
| PPO | 59.3% | 63.8% | **+4.6 pp** |
| SAC | 79.8% | 74.9% | **−4.9 pp** |

**Note:** All 20 test days were classified as "Mild" (avg temp 19–25°C, wind < 3 m/s), so the ablation captures only mild weather variation. More extreme conditions (hot, cold, windy) were not present in the test period.

---

## 8. DQN Strategy Selection — Chapter 4: "What Did the Agent Learn?"

![DQN Timeline](stage2/output/caltech_rl_dqn_timeline.png)

### Strategy Usage Frequency

| Strategy | Count | Percentage |
|----------|------:|:----------:|
| Round-Robin | 1,398 | 97.1% |
| Uncontrolled | 36 | 2.5% |
| Conservative | 6 | 0.4% |

---

## 9. Key Findings

### Research Question 1: _"How far are heuristics from optimal?"_ — The Baseline Gap

**Finding: Heuristics nearly match the MPC Oracle on satisfaction, but fail catastrophically on peak demand.**

All four heuristic baselines (Uncontrolled, FCFS, EDF, Round-Robin) achieve 95.5–96.2% satisfaction — within 1 percentage point of the MPC Oracle's 96.2%. The reason is that the Caltech site is only moderately constrained: most days have enough capacity to serve all EVs.

However, the baselines produce **2–2.7× higher peak demand** than the MPC Oracle:
- MPC Oracle: **53.7 kW** (optimally smoothed)
- Round-Robin: 105.4 kW (+96%)
- FCFS: 114.1 kW (+112%)
- Uncontrolled: 144.9 kW (+170%)

This confirms that the optimization opportunity lies not in _how much_ energy is delivered, but in _when_ it is delivered — i.e., **load shaping**. Stage 1's finding that 38% of sessions are "park-and-forget" (Cluster C4: long idle, low utilization) reinforces this: there is enormous temporal flexibility for smart scheduling.

### Research Question 2: _"Can model-free RL close that gap?"_ — Learning to Schedule

**Finding: DQN (meta-strategy) matches the best heuristics; continuous-action RL agents struggle with the high-dimensional action space.**

- **DQN achieves 95.5% satisfaction** — matching Round-Robin exactly. This is because DQN learned to select Round-Robin 97% of the time (see Section 8), effectively discovering through trial-and-error that Round-Robin is the best heuristic for this environment. This is a non-trivial result: the agent explored all 5 strategies and converged on the optimal one.
- **SAC achieves 79.8% satisfaction** — the best continuous-action agent, but still ~16 pp below heuristics. The 54-dimensional continuous action space (one rate per station) is difficult to optimize with 60K timesteps.
- **PPO and DDPG significantly underperform** (59.3% and 71.9% respectively). PPO's on-policy nature wastes samples in this high-dimensional setting. DDPG's deterministic policy with noise exploration struggles to discover cooperative multi-station strategies.
- **Peak demand**: Interestingly, PPO achieves the _lowest_ peak demand (65.5 kW) among all RL agents — close to MPC Oracle's 53.7 kW. This suggests PPO learned conservative, peak-shaving behavior even though it sacrificed satisfaction to do so. This reveals a fundamental **satisfaction–peak demand tradeoff** in the reward function.

**Takeaway**: With only 60K training steps, discrete meta-action RL (DQN) is far more practical than continuous-action RL for this 54-station environment. Continuous agents would likely need 10–100× more training to converge.

### Research Question 3: _"Does weather context improve scheduling?"_ — Weather Impact

**Finding: Weather provides a modest boost to PPO (+4.6 pp) but slightly hurts SAC (−4.9 pp). The effect is inconclusive due to limited weather variation in the test period.**

- **PPO +Weather**: 63.8% vs 59.3% baseline (+4.6 pp). The additional 5 weather features (temperature, humidity, wind speed, visibility, precipitation) appear to help PPO's policy network discover useful context. This may be because weather features act as a regularizing signal, providing temporal structure that helps PPO's limited exploration.
- **SAC +Weather**: 74.9% vs 79.8% baseline (−4.9 pp). The expanded observation space (54×13 instead of 54×8) may have introduced noise, slowing SAC's convergence within the fixed training budget. SAC's entropy-maximizing objective may spread exploration over the larger observation space without gaining useful information.
- **Context**: All 20 test days fell in the "Mild" weather category (19–25°C, low wind). Stage 1 found that weather has **weak direct correlation** with charging behavior (r ≈ 0), with effects limited to indirect interactions (e.g., hot evenings → longer sessions). The mild test period means the weather features provided minimal discriminative signal.
- **Connection to Stage 1**: This result is **consistent with Stage 1's finding** that weather contributes only ~2–3% R² improvement in prediction models. Weather matters at the margins (extreme conditions), not in the average case.

### Research Question 4: _"What did the agents learn?"_ — Connecting to Stage 1

**Finding: DQN independently discovered that Round-Robin is the optimal heuristic, switching to Uncontrolled only during low-demand overnight hours.**

The DQN strategy timeline reveals two clear behavioral patterns:

1. **Round-Robin dominance (97.1%)**: DQN selects Round-Robin for nearly all timesteps, especially during the peak demand period (14:00–18:00) when Stage 1 identified the highest occupancy. Round-Robin distributes power fairly across stations — echoing Stage 1's finding that the largest user cluster (C4, "workplace parkers") parks for long hours, giving the scheduler flexibility to spread charging load over time.

2. **Uncontrolled during low demand (2.5%)**: DQN switches to Uncontrolled (charge everyone at max rate) during the overnight hours (0:00–2:00 AM) when few EVs are present and capacity constraints don't bind. This mirrors Stage 1's temporal pattern: very few sessions start between midnight and 5 AM, so aggressive charging is safe.

3. **Conservative strategy is rarely used (0.4%)**: The agent learned that charging at 50% rate is almost never optimal — it's better to either distribute fairly (Round-Robin) or charge aggressively (Uncontrolled) than to throttle.

**Connection to Stage 1 behavioral clusters:**
- The DQN agent's learned policy implicitly handles both of Stage 1's major user segments: _quick top-up users_ (Cluster C0, high utilization) get immediate service via Round-Robin, while _workplace parkers_ (Cluster C4, low utilization, long idle time) get their energy spread over their long parking duration.
- The transition from Uncontrolled to Round-Robin as demand increases mirrors the bimodal arrival pattern discovered in Stage 1's EDA: low overnight activity followed by the 7–9 AM arrival wave.

### Overall Assessment

| Criterion | Best Policy | Key Metric |
|-----------|------------|------------|
| **Satisfaction** | MPC Oracle / Uncontrolled | 96.2% |
| **Peak Shaving** | MPC Oracle | 53.7 kW |
| **Fairness** | EDF / Uncontrolled | 0.998 |
| **Best RL Agent** | DQN (meta) | 95.5% / 105.4 kW |
| **Best Continuous RL** | SAC | 79.8% / 108.8 kW |
| **Fastest Training** | DQN | 3.8 min |

**The main conclusion of Stage 2 is that the EV charging scheduling problem at Caltech is more about _when_ to deliver energy (load shaping) than _whether_ to deliver it.** Heuristics already achieve near-optimal satisfaction because most days have sufficient capacity. The true value of intelligent scheduling — exemplified by the MPC Oracle's 53.7 kW peak vs heuristics' 105–145 kW — lies in peak demand reduction, which continuous RL agents have begun to learn (PPO's 65.5 kW peak) but need significantly more training to master.

**Future directions:**
1. **Scale training**: 500K–1M timesteps for SAC/PPO to allow convergence in the continuous action space
2. **Reward tuning**: Increase peak demand penalty weight (β) to encourage more aggressive load shaping
3. **Constrained RL**: Add hard infrastructure constraints (transformer limits) to the environment
4. **Seasonal evaluation**: Test on extreme weather days to properly assess weather integration value
5. **Multi-site transfer**: Train on Caltech, evaluate on JPL to test generalization across site topologies

---

## 10. Files & Reproducibility

### Source Code
| File | Description |
|------|-------------|
| `stage2/ev_charging_env.py` | Base Gymnasium environment wrapping ACN-Sim |
| `stage2/weather_env.py` | Weather-augmented environment (+5 climate features) |
| `stage2/discrete_env.py` | Discrete meta-strategy wrapper for DQN |
| `stage2/rl_agents.py` | Stable-Baselines3 agent factory |
| `stage2/train_rl.py` | Training loop with episode metrics callback |
| `stage2/evaluate_rl.py` | Unified evaluation (RL + baselines + MPC Oracle) |
| `stage2/analyze_rl.py` | Policy analysis & Stage 1 connections |
| `stage2/visualize_rl.py` | 6 publication-quality visualization functions |
| `stage2/run_experiments.py` | Full experiment orchestrator |

### Output Files (`stage2/output/`)
- 6 RL plots (`caltech_rl_*.png`)
- 2 evaluation CSVs (details + summary)
- 3 analysis CSVs (hourly actions, DQN timeline, weather ablation)

### Trained Models (`stage2/models/`)
- 6 model directories with saved SB3 checkpoints, training metrics, and config files

### How to Run
```bash
python -m stage2.run_experiments --site caltech --timesteps 60000
```
Total training time: ~16.5 hours (dominated by DDPG). Evaluation + analysis + visualization: ~7 minutes.
