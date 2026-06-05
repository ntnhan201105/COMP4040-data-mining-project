"""Factory module for creating Stable-Baselines3 RL agents.

Provides ``create_agent(algo, env)`` with tuned hyperparameters for each of
DQN, PPO, DDPG, and SAC, ensuring a fair comparison (same network architecture,
compatible training budgets).
"""
from typing import Optional

import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, DQN
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# ── Default configs per algorithm ────────────────────────────────────────
AGENT_CONFIGS = {
    "PPO": {
        "class": PPO,
        "policy": "MlpPolicy",
        "kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 288,          # 1 full episode per rollout
            "batch_size": 64,
            "n_epochs": 10,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        },
    },
    "SAC": {
        "class": SAC,
        "policy": "MlpPolicy",
        "kwargs": {
            "learning_rate": 3e-4,
            "buffer_size": 50_000,
            "learning_starts": 500,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    },
    "DDPG": {
        "class": DDPG,
        "policy": "MlpPolicy",
        "kwargs": {
            "learning_rate": 1e-3,
            "buffer_size": 50_000,
            "learning_starts": 500,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    },
    "DQN": {
        "class": DQN,
        "policy": "MlpPolicy",
        "kwargs": {
            "learning_rate": 1e-3,
            "buffer_size": 50_000,
            "learning_starts": 500,
            "batch_size": 64,
            "gamma": 0.99,
            "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "target_update_interval": 500,
        },
    },
}


def create_agent(
    algo: str,
    env,
    seed: int = 42,
    tensorboard_log: Optional[str] = None,
    verbose: int = 0,
    **override_kwargs,
):
    """Create a configured SB3 agent.

    Args:
        algo: Algorithm name — ``"PPO"``, ``"SAC"``, ``"DDPG"``, or ``"DQN"``.
        env: Gymnasium or VecEnv environment.
        seed: Random seed for reproducibility.
        tensorboard_log: Optional directory for TensorBoard logs.
        verbose: SB3 verbosity level.
        **override_kwargs: Override any default hyperparameter.

    Returns:
        Configured (untrained) SB3 model instance.
    """
    algo = algo.upper()
    if algo not in AGENT_CONFIGS:
        raise ValueError(
            f"Unknown algorithm '{algo}'. Choose from {list(AGENT_CONFIGS)}"
        )

    config = AGENT_CONFIGS[algo]
    cls = config["class"]
    policy = config["policy"]
    kwargs = {**config["kwargs"], **override_kwargs}

    # Shared network architecture for fair comparison
    policy_kwargs = kwargs.pop("policy_kwargs", {})
    policy_kwargs.setdefault("net_arch", [256, 256])

    # Exploration noise for DDPG
    if algo == "DDPG":
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else 1
        kwargs.setdefault(
            "action_noise",
            OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim),
                sigma=0.1 * np.ones(action_dim),
            ),
        )

    return cls(
        policy,
        env,
        seed=seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        **kwargs,
    )
