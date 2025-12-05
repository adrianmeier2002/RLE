import gymnasium as gym
import numpy as np
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import imageio

def run_random_agent(
        episodes: int = 100,
        render: bool = False,
        video_every: int = None,
        seeds: int = None
):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")

    run_name = f"baseline_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")

    rewards = []

    for ep in range(episodes):
        if seeds is not None:
            obs, info =env.reset(seed=seeds + ep)
        else:
            obs, info = env.reset()     # Initialize the environment

        done = False                # Done means episode ended naturally
        truncated = False           # Trunication means episode ended due to time limit
        ep_reward = 0               # Initialize episode reward

        frames = []          # For storing frames if rendering

        while not (done or truncated):
            action = env.action_space.sample()  # Sample random action
            next_obs, reward, done, truncated, info = env.step(action)  # Take action in the environment
            ep_reward += reward  # Accumulate reward

            if video_every and (ep % video_every == 0):
                frame = env.render()
                frames.append(frame)

        rewards.append(ep_reward)
        writer.add_scalar("Reward/Episode", ep_reward, ep)

        print(f"Episode {ep + 1}/{episodes} - Reward: {ep_reward}")

        if video_every and (ep % video_every == 0):
            save_video(frames, f"videos/baseline/episode_{ep}.mp4")

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

def save_video(frames, path, fps=30):
    imageio.mimsave(path, frames, fps=fps)

if __name__ == "__main__":
    run_random_agent(episodes=100, render=False, video_every=20, seeds=42)