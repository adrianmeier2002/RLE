import gymnasium as gym
import os
import numpy as np
import ale_py

def make_env(env_id: str = "ALE/SpaceInvaders-v5", record_video: bool = False, video_folder: str = "videos/", video_freq: int = 100):
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

    # Create base environment
    env = gym.make(
        env_id,
        frameskip=4,               # Built-in frameskipping
        repeat_action_probability=0, # No sticky actions
        render_mode='rgb_array' if record_video else None
    )

    # Video recording wrapper
    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda episode_id: episode_id % video_freq == 0
        )

    return env

# Helper function for evaluation (no wrappers)
def make_eval_env(env_id: str = "ALE/SpaceInvaders-v5"):
    """
    Creates a raw Atari environment for evaluation.
    Deterministic behaviour without any preprocessing.
    """
    env = gym.make(
        env_id,
        frameskip=4,
        repeat_action_probability=0
    )

    return env

# Example usage
if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    env.close()
