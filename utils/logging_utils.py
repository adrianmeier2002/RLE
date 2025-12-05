import os
from torch.utils.tensorboard import SummaryWriter
import imageio

def create_writer(log_dir: str, run_name: str):
    """Creates a TensorBoard writer for logging metrics."""
    path = os.path.join(log_dir, run_name)
    os.makedirs(path, exist_ok=True)
    writer = SummaryWriter(path)
    return writer

def log_scalar(writer, tag: str, value, step: int):
    """Logs a scalar value to TensorBoard."""
    if writer is not None:
        writer.add_scalar(tag, value, step)

def save_video(frames, path, fps=30):
    """Saves a list of frames as a video file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, format="FFMPEG", fps=fps)

def save_model(agent, path: str):
    """Saves a PyTorch model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(agent.state_dict(), path)