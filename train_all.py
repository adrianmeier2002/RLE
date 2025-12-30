"""
Master Training Script - Train all DQN variants sequentially
Run this once and let it train all agents overnight!
"""

import os
import time
import json
import traceback
from datetime import datetime

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


def train_agent_safe(config):
    """
    Train a single agent with error handling.
    
    Parameters
    ----------
    config : dict
        Training configuration
        
    Returns
    -------
    success : bool
    duration : float
    error : str or None
    """
    start_time = time.time()
    
    try:
        print(f"\n{'='*80}")
        print(f"STARTING: {config['name']}")
        print(f"{'='*80}")
        print(f"Agent Class: {config['agent'].__name__}")
        print(f"Buffer: {config['buffer'].__name__}")
        print(f"Steps: {config['num_steps']:,}")
        print(f"Model Path: {config['model_save']}")
        print(f"{'='*80}\n")
        
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
            model_save=config['model_save']
        )
        
        duration = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"SUCCESS: {config['name']}")
        print(f"Duration: {format_time(duration)}")
        print(f"Model saved: {config['model_save']}")
        print(f"{'='*80}\n")
        
        return True, duration, None
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        
        print(f"\n{'='*80}")
        print(f"!! FAILED: {config['name']} !!")
        print(f"Duration: {format_time(duration)}")
        print(f"Error: {error_msg}")
        print(f"{'='*80}")
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*80}\n")
        
        return False, duration, error_msg


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
    
    # Training summary
    results_summary = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_agents': len(training_configs),
        'agents': []
    }
    
    total_start = time.time()
    successful = 0
    failed = 0
    
    print("\n" + "="*80)
    print("MASTER TRAINING SCRIPT - ALL DQN VARIANTS")
    print("="*80)
    print(f"Total agents to train: {len(training_configs)}")
    print(f"Steps per agent: {common_params['num_steps']:,}")
    print(f"Start time: {results_summary['start_time']}")
    print("="*80)
    
    # Train each agent
    for i, config in enumerate(training_configs, 1):
        print(f"\n{'#'*80}")
        print(f"AGENT {i}/{len(training_configs)}")
        print(f"{'#'*80}")
        
        success, duration, error = train_agent_safe(config)
        
        # Record results
        agent_result = {
            'name': config['name'],
            'success': success,
            'duration_seconds': duration,
            'duration_formatted': format_time(duration),
            'model_path': config['model_save'],
            'error': error
        }
        results_summary['agents'].append(agent_result)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Save intermediate results
        results_summary['successful'] = successful
        results_summary['failed'] = failed
        results_summary['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open('results/training_progress.json', 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        # Print progress
        remaining = len(training_configs) - i
        print(f"\n{'='*80}")
        print(f"PROGRESS: {i}/{len(training_configs)} agents completed")
        print(f"Successful: {successful} | Failed: {failed} | Remaining: {remaining}")
        print(f"{'='*80}\n")
    
    # Final summary
    total_duration = time.time() - total_start
    results_summary['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_summary['total_duration_seconds'] = total_duration
    results_summary['total_duration_formatted'] = format_time(total_duration)
    results_summary['successful'] = successful
    results_summary['failed'] = failed
    
    # Save final results
    with open('results/training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Total Duration: {format_time(total_duration)}")
    print(f"Successful: {successful}/{len(training_configs)}")
    print(f"Failed: {failed}/{len(training_configs)}")
    print("\nResults saved to:")
    print("  - results/training_summary.json")
    print("  - results/training_progress.json")
    print("="*80)
    
    # Print per-agent summary
    print("\n" + "="*80)
    print("AGENT TRAINING TIMES")
    print("="*80)
    for agent in results_summary['agents']:
        status = "SUCCESS" if agent['success'] else "!! FAILED !!"
        print(f"{status} {agent['name']:20s} | {agent['duration_formatted']}")
    print("="*80 + "\n")
    
    if failed > 0:
        print("\n!!  Some agents failed to train. Check results/training_summary.json for details. !!\n")
    else:
        print("\nAll agents trained successfully!\n")
        print("Next steps:")
        print("  1. Run evaluation: python -m utils.evaluation")
        print("  2. Check TensorBoard: tensorboard --logdir=runs/")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("!! TRAINING INTERRUPTED BY USER !!")
        print("="*80)
        print("Partial results saved in results/training_progress.json")
        print("You can resume by commenting out completed agents in the script.")
        print("="*80 + "\n")
    except Exception as e:
        print("\n\n" + "="*80)
        print("!! CRITICAL ERROR IN MASTER SCRIPT !!")
        print("="*80)
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*80 + "\n")