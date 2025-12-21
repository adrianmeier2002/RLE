import gymnasium as gym
import os
import numpy as np
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from collections import deque

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        old_shape = env.observation_space.shape
        new_shape = (old_shape[0] * k,) + old_shape[1:]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info
    
    def _get_obs(self):
        assert len(self.frames) == self.k, f"Expected {self.k} frames, got {len(self.frames)}"
        frames_list = list(self.frames)

        if frames_list[0].ndim == 2:  # Shape is (H, W)
            # Stack to get (k, H, W)
            return np.stack(frames_list, axis=0)
        else:  # Shape is (C, H, W)
            # Concatenate along channel dimension
            return np.concatenate(frames_list, axis=0)

def make_env(
        env_id: str = "ALE/SpaceInvaders-v5", 
        eval_mode: bool = False,
        record_video: bool = False, 
        video_folder: str = "videos/", 
        video_freq: int = 100,
        frame_size: int = 84,
        frameskip: int = 4,
        num_stack: int = 4
):
    """
    Creates a basic Atari environment without any wrappers.
    Useful for testing or baseline agents.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.

    Returns
    -------
    env : gym.Env
        Raw environment.
    """

    render_mode = 'rgb_array' if (record_video and not eval_mode) else None

    # Create base environment
    env = gym.make(
        env_id,
        frameskip=frameskip,               # Built-in frameskipping
        repeat_action_probability=0, # No sticky actions
        render_mode=render_mode
    )

    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (frame_size, frame_size))

    env = FrameStack(env, num_stack)

    # Video recording wrapper
    if record_video and not eval_mode:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda ep: ep % video_freq == 0
        )

    return env

# Test the environment
if __name__ == "__main__":
    env = make_env(num_stack=4)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Expected shape: (4, 84, 84)")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: obs shape = {obs.shape}, reward = {reward}")
        if done or truncated:
            obs, info = env.reset()
            print("Environment reset")
    
    env.close()
    print("Test completed successfully!")