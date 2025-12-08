import gymnasium as gym
import os
import numpy as np
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

def make_env(
        env_id: str = "ALE/SpaceInvaders-v5", 
        eval_mode: bool = False,
        record_video: bool = False, 
        video_folder: str = "videos/", 
        video_freq: int = 100,
        frame_size: int = 84
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
        frameskip=4,               # Built-in frameskipping
        repeat_action_probability=0, # No sticky actions
        render_mode=render_mode
    )

    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (frame_size, frame_size))

    # Video recording wrapper
    if record_video and not eval_mode:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda ep: ep % video_freq == 0
        )

    return env

# Example usage
if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    env.close()
