import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import imageio
import ale_py
import os

from utils.env_utils import make_env
from utils.logging_utils import create_writer, log_scalar

def run_random_agent(
        episodes: int = 100,
        video_every: int = None,
        seeds: int = None
):
    """
    Runs a baseline random agent in the Atari environment for a specified number of episodes.
    """

    env = make_env(
        record_video=bool(video_every),
        video_folder="videos/baseline/",
        video_freq=video_every
    )

    run_name = f"baseline_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = create_writer("runs/baseline_random", run_name)

    rewards = []

    for ep in range(episodes):
        if seeds is not None:
            obs, info =env.reset(seed=seeds + ep)
        else:
            obs, info = env.reset()     # Initialize the environment

        done = False                # Done means episode ended naturally
        truncated = False           # Trunication means episode ended due to time limit
        ep_reward = 0               # Initialize episode reward

        while not (done or truncated):
            action = env.action_space.sample()  # Sample random action
            next_obs, reward, done, truncated, info = env.step(action)  # Take action in the environment
            ep_reward += reward  # Accumulate reward

        rewards.append(ep_reward)
        log_scalar(writer, "Reward/Episode", ep_reward, ep)

        print(f"Episode {ep + 1}/{episodes} - Reward: {ep_reward}")

    env.close()

    print("\n" + "="*60)
    print("Baseline Random Agent Results")
    print(f"Mean Reward: {np.mean(rewards):.2f}")
    print(f"Median Reward: {np.median(rewards):.2f}")
    print(f"Std: {np.std(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print("="*60 + "\n")

    return rewards

if __name__ == "__main__":
    run_random_agent(episodes=1000, render=False, video_every=100, seeds=42)