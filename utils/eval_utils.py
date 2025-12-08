import os
import numpy as np
from datetime import datetime
import imageio
from utils.env_utils import make_env

def evaluate_agent(
        agent,
        episodes: int = 100,
        render: bool = False,
        record_video: bool = False,
        video_folder: str = "videos/eval/",
        video_every: int = 50,
        seeds: int = None
):
    """Evaluate a given agent in the Atari environment over a number of episodes and saves the metrics and optional videos."""

    env = make_env(eval_mode=True)
    rewards = []

    for ep in range(episodes):
        if seeds is not None:
            obs, info = env.reset(seed=seeds + ep)
        else:
            obs, info = env.reset()

        done = False
        truncated = False
        ep_reward = 0

        while not (done or truncated):
            action = agent.select_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            if render:
                env.render()

        rewards.append(ep_reward)

        print(f"Episode {ep+1}/{episodes} - Reward: {ep_reward}")

    env.close()


    print("\n" + "="*60)
    print(f"Evaluation Results over {episodes} Episodes")
    print(f"Mean Reward: {np.mean(rewards):.2f}")
    print(f"Std Reward: {np.std(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print("="*60 + "\n")

    return rewards