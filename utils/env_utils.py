import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
import os

def make_env(
        env_id: str = "ALE/SpaceInvaders-v5",
        frame_skip: int = 4,
        frame_stack: int = 4,
        grayscale: bool = True,
        resize: bool = True,
        width: int = 84,
        height: int = 84,
        record_video: bool = False,
        video_folder: str = "videos/",
        video_frequency: int = 5000
):
    """
    Creates a consistent Atari environment with preprocessing.
    Compatible with DQN and other value-based methods.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    frame_skip : int
        How many frames to repeat each chosen action.
    frame_stack : int
        Number of consecutive frames to stack.
    grayscale : bool
        Whether to convert frames to greyscale.
    resize : bool
        Resize frames to (height, width)
    width : int
        Width for resize.
    height : int
        Height for resize.
    record_video : bool
        Record a video every N steps.
    video_folder : str
        Where to store the videos.
    video_frequency : int
        Record video at every N env steps.
    seed : int
        Random seed for environment for reproducibility.
    
    Returns
    -------
    env : gym.Env
        Preprocessed environment ready for RL training.
    """

    # Create base environment
    env = gym.make(
        env_id,
        frameskip=frame_skip,       # built-in frameskipping
        repeat_action_probability=0 # No sticky actions
    )

    # Preprocessing
    if grayscale:
        env = GrayScaleObservation(env)
    
    if resize:
        env = ResizeObservation(env, shape=(height, width))

    env = FrameStack(env, frame_stack)

    if record_video:
        # Ensure video folder exists
        os.makedirs(video_folder, exist_ok=True)

        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep: ep % video_frequency == 0
        )

    return env


# Helper funcition for evaluation (no frameskip modification)
def make_eval_env(
        env_id: str = "ALE/SpaceInvaders-v5",
        grayscale: bool = True,
        resize: bool = True,
        width: int = 84,
        height: int = 84,
        frame_stack: int = 4
):
    """Environment for evaluating a trained agent.
       No video recording, deterministic behaviour."""
    
    env = gym.make(
        env_id,
        frameskip=4,
        repeat_action_probability=0
    )

    if grayscale:
        env = GrayScaleObservation(env)

    if resize:
        env = ResizeObservation(env, shape=(height, width))

    env = FrameStack(env, frame_stack)

    return env