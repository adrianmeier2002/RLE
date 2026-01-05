"""
Master Training Script - Train all DQN variants sequentially
Run this once and let it train all agents overnight!
"""

import os
import time
import json
import traceback
from datetime import datetime
from multiprocessing import Process, Queue, cpu_count, set_start_method
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from dqn.dqn_initial import DQNAgent, train_dqn
from dqn.dqn_double import DoubleDQNAgent
from dqn.dqn_dueling import DuelingDQNAgent
from dqn.dqn_per_buffer import PerDQNAgent
from dqn.dqn_noisy import NoisyDQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.replay_buffer_per import PrioritizedReplayBuffer


def format_time(seconds):
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def train_agent_worker(config, result_queue):
    """
    Worker function for parallel training.
    Runs in separate process.
    """
    start_time = time.time()

    try:
        # Force each worker to use specific GPU (optional)
        if 'gpu_id' in config:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])
        
        # Clear CUDA cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n[WORKER {os.getpid()}] Starting: {config['name']}")
        
        # Train the agent
        agent, rewards = train_dqn(
            env_id=config['env_id'],
            agent=config['agent'],
            buffer_class=config['buffer'],
            num_steps=config['num_steps'],
            batch_size=config['batch_size'],
            target_update_freq=config['target_update_freq'],
            learning_starts=config['learning_starts'],
            train_freq=config['train_freq'],
            video_every=config['video_every'],
            video_folder=config['video_folder'],
            writer_path=config['writer_path'],
            model_save=config['model_save'],
            eval_freq=config['eval_freq'],
            eval_episodes=config['eval_episodes']
        )
        
        # Clear CUDA cache after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        duration = time.time() - start_time
        
        result = {
            'name': config['name'],
            'success': True,
            'duration': duration,
            'final_reward': float(rewards[-1]) if len(rewards) > 0 else 0.0,
            'error': None
        }
        
        result_queue.put(result)
        print(f"[WORKER {os.getpid()}] ✓ Completed: {config['name']} in {format_time(duration)}")
        
    except Exception as e:
        duration = time.time() - start_time
        error_trace = traceback.format_exc()
        
        result = {
            'name': config['name'],
            'success': False,
            'duration': duration,
            'final_reward': 0.0,
            'error': str(e),
            'traceback': error_trace
        }
        
        result_queue.put(result)
        print(f"[WORKER {os.getpid()}] ✗ Failed: {config['name']}")
        print(f"Error: {str(e)}")
        print(error_trace)

def train_parallel(training_configs, max_parallel=4):
    """
    Train agents in parallel using multiprocessing.
    
    Parameters
    ----------
    training_configs : list
        List of training configurations
    max_parallel : int
        Maximum number of agents to train simultaneously
    """
    results_summary = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_agents': len(training_configs),
        'max_parallel': max_parallel,
        'agents': []
    }
    
    total_start = time.time()
    result_queue = Queue()
    
    print("\n" + "="*80)
    print("PARALLEL TRAINING - ALL DQN VARIANTS")
    print("="*80)
    print(f"Total agents: {len(training_configs)}")
    print(f"Parallel workers: {max_parallel}")
    print(f"Start time: {results_summary['start_time']}")
    print("="*80 + "\n")
    
    # Process agents in batches
    for i in range(0, len(training_configs), max_parallel):
        batch = training_configs[i:i + max_parallel]
        processes = []
        
        print(f"\n{'#'*80}")
        print(f"STARTING BATCH {i//max_parallel + 1}")
        print(f"{'#'*80}")
        
        # Start processes
        for config in batch:
            p = Process(target=train_agent_worker, args=(config, result_queue))
            p.start()
            processes.append(p)
            print(f"Started: {config['name']}")

        print(f"\nWaiting for {len(processes)} workers to complete...\n")
        
        # Wait for all processes in batch to complete
        for p in processes:
            p.join()
        
        # Collect results
        for _ in range(len(batch)):
            result = result_queue.get()
            results_summary['agents'].append(result)
            
            # Save intermediate progress
            os.makedirs('results', exist_ok=True)
            with open('results/training_progress.json', 'w') as f:
                json.dump(results_summary, f, indent=4)
    
    # Final summary
    total_duration = time.time() - total_start
    results_summary['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_summary['total_duration_seconds'] = total_duration
    results_summary['total_duration_formatted'] = format_time(total_duration)
    
    successful = sum(1 for a in results_summary['agents'] if a['success'])
    failed = len(results_summary['agents']) - successful
    
    results_summary['successful'] = successful
    results_summary['failed'] = failed
    
    with open('results/training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # Print summary
    print("\n" + "="*80)
    print("PARALLEL TRAINING COMPLETE!")
    print("="*80)
    print(f"Total Duration: {format_time(total_duration)}")
    print(f"Successful: {successful}/{len(training_configs)}")
    print(f"Failed: {failed}/{len(training_configs)}")
    print("="*80 + "\n")

    print("Individual Results:")
    print("="*80)
    for agent in results_summary['agents']:
        status = "✓" if agent['success'] else "✗"
        print(f"{status} {agent['name']:20s} | {format_time(agent['duration'])}")
    print("="*80 + "\n")


def main():
    """Train all DQN variants sequentially."""
    
    # Common training parameters
    common_params = {
        'env_id': 'ALE/SpaceInvaders-v5',
        'num_steps': 5_000_000,
        'batch_size': 32,
        'target_update_freq': 10_000,
        'learning_starts': 50_000,
        'train_freq': 4,
        'video_every': 100,
        'eval_freq': 100000,
        'eval_episodes': 10
    }
    
    # Configure all agents
    training_configs = [
        {
            'name': 'DQN (Baseline)',
            'agent': DQNAgent,
            'buffer': ReplayBuffer,
            'video_folder': 'videos/dqn_initial/',
            'writer_path': 'runs/dqn_initial',
            'model_save': 'dqn/models/dqn_initial.pt',
            **common_params
        },
        {
            'name': 'Double DQN',
            'agent': DoubleDQNAgent,
            'buffer': ReplayBuffer,
            'video_folder': 'videos/dqn_double/',
            'writer_path': 'runs/dqn_double',
            'model_save': 'dqn/models/dqn_double.pt',
            **common_params
        },
        {
            'name': 'Dueling DQN',
            'agent': DuelingDQNAgent,
            'buffer': ReplayBuffer,
            'video_folder': 'videos/dqn_dueling/',
            'writer_path': 'runs/dqn_dueling',
            'model_save': 'dqn/models/dqn_dueling.pt',
            **common_params
        },
        {
            'name': 'PER DQN',
            'agent': PerDQNAgent,
            'buffer': PrioritizedReplayBuffer,
            'video_folder': 'videos/dqn_per/',
            'writer_path': 'runs/dqn_per',
            'model_save': 'dqn/models/dqn_per.pt',
            **common_params
        },
        {
            'name': 'Noisy DQN',
            'agent': NoisyDQNAgent,
            'buffer': ReplayBuffer,
            'video_folder': 'videos/dqn_noisy/',
            'writer_path': 'runs/dqn_noisy',
            'model_save': 'dqn/models/dqn_noisy.pt',
            **common_params
        },
    ]
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('dqn/models', exist_ok=True)
    
    # Determine parallel workers
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        if gpu_memory >= 6:  # 6GB+ GPU
            max_parallel = 2  # 2 Agenten gleichzeitig
            print(f"GPU detected ({gpu_memory:.1f}GB). Training 2 agents in parallel.")
        else:
            max_parallel = 1
            print(f"GPU detected ({gpu_memory:.1f}GB). Training 1 agent at a time.")
    else:
        max_parallel = max(1, cpu_count() // 4)
        print(f"CPU training. Using {max_parallel} parallel workers.")
    # Run parallel training
    train_parallel(training_configs, max_parallel=max_parallel)


if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)

    except RuntimeError:
        pass

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("!! TRAINING INTERRUPTED BY USER !!")
        print("="*80)
    except Exception as e:
        print("\n\n" + "="*80)
        print("!! CRITICAL ERROR IN MASTER SCRIPT !!")
        print("="*80)
        print(f"Error: {e}")
        traceback.print_exc()
        print("="*80 + "\n")