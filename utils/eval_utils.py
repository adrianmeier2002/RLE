import os
import numpy as np
import torch
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from utils.env_utils import make_env


def load_agent(agent_class, model_path: str, input_shape: tuple, num_actions: int):
    """
    Load a trained agent from a checkpoint.
    
    Parameters
    ----------
    agent_class : class
        The agent class (e.g., DQNAgent, DoubleDQNAgent)
    model_path : str
        Path to the saved model weights
    input_shape : tuple
        Shape of input observations (C, H, W)
    num_actions : int
        Number of possible actions
        
    Returns
    -------
    agent : Agent
        Loaded agent ready for evaluation
    """
    agent = agent_class(input_shape, num_actions)
    agent.q_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.q_net.eval()  # Set to evaluation mode
    print(f"Loaded model from {model_path}")
    return agent


def evaluate_agent(
        agent,
        env_id: str = "ALE/SpaceInvaders-v5",
        num_episodes: int = 100,
        render: bool = False,
        record_video: bool = False,
        video_folder: str = "videos/eval/",
        video_freq: int = 10,
        seed: Optional[int] = None,
        deterministic: bool = True,
        save_results: bool = True,
        results_folder: str = "results/",
        agent_name: str = "dqn"
) -> Dict:
    """
    Evaluate a trained agent over multiple episodes.
    
    Parameters
    ----------
    agent : Agent
        The trained agent to evaluate
    env_id : str
        Gymnasium environment ID
    num_episodes : int
        Number of episodes to run
    render : bool
        Whether to render the environment (slow)
    record_video : bool
        Whether to record videos
    video_folder : str
        Folder to save videos
    video_freq : int
        Record every N episodes
    seed : int, optional
        Random seed for reproducibility
    deterministic : bool
        If True, use greedy action selection (no exploration)
    save_results : bool
        Whether to save results to JSON
    results_folder : str
        Folder to save results
    agent_name : str
        Name of the agent for saving results
        
    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics
    """
    
    # Create environment
    env = make_env(
        env_id=env_id,
        eval_mode=True,
        record_video=record_video,
        video_folder=video_folder,
        video_freq=video_freq
    )
    
    # Set agent to evaluation mode
    if hasattr(agent, 'q_net'):
        agent.q_net.eval()
    
    # Storage for metrics
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n{'='*70}")
    print(f"Evaluating {agent_name.upper()} for {num_episodes} episodes")
    print(f"Environment: {env_id}")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*70}\n")
    
    for ep in range(num_episodes):
        # Reset environment with seed
        if seed is not None:
            obs, info = env.reset(seed=seed + ep)
        else:
            obs, info = env.reset()
        
        done = False
        truncated = False
        ep_reward = 0
        ep_length = 0
        
        while not (done or truncated):
            # Select action (deterministic or with exploration)
            if deterministic and hasattr(agent, 'select_action'):
                # Try to use eval_mode if available
                if 'eval_mode' in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(obs, eval_mode=True)
                else:
                    action = agent.select_action(obs)
            else:
                action = agent.select_action(obs)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        # Print progress
        if (ep + 1) % 10 == 0:
            mean_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {ep+1}/{num_episodes} | "
                  f"Last 10 Avg: {mean_reward:.2f} | "
                  f"Current: {ep_reward:.2f}")
    
    env.close()
    
    # Calculate statistics
    results = {
        'agent_name': agent_name,
        'env_id': env_id,
        'num_episodes': num_episodes,
        'seed': seed,
        'deterministic': deterministic,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'median_reward': float(np.median(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS: {agent_name.upper()}")
    print(f"{'='*70}")
    print(f"Episodes:        {num_episodes}")
    print(f"Mean Reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Median Reward:   {results['median_reward']:.2f}")
    print(f"Min Reward:      {results['min_reward']:.2f}")
    print(f"Max Reward:      {results['max_reward']:.2f}")
    print(f"Mean Length:     {results['mean_length']:.2f} steps")
    print(f"{'='*70}\n")
    
    # Save results
    if save_results:
        os.makedirs(results_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_path = os.path.join(results_folder, f'{agent_name}_eval_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {json_path}")
        
        # Save plot
        plot_path = os.path.join(results_folder, f'{agent_name}_eval_{timestamp}.png')
        plot_evaluation_results(results, save_path=plot_path)
        print(f"Plot saved to {plot_path}")
    
    return results


def plot_evaluation_results(results: Dict, save_path: Optional[str] = None):
    """
    Create visualization of evaluation results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_agent
    save_path : str, optional
        If provided, save figure to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Evaluation Results: {results['agent_name'].upper()}", 
                 fontsize=16, fontweight='bold')
    
    rewards = results['episode_rewards']
    lengths = results['episode_lengths']
    
    # 1. Episode rewards over time
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.6, linewidth=1)
    ax1.plot(np.convolve(rewards, np.ones(10)/10, mode='valid'), 
             color='red', linewidth=2, label='10-episode MA')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards Over Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward distribution
    ax2 = axes[0, 1]
    ax2.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(results['mean_reward'], color='red', 
                linestyle='--', linewidth=2, label=f"Mean: {results['mean_reward']:.1f}")
    ax2.axvline(results['median_reward'], color='green', 
                linestyle='--', linewidth=2, label=f"Median: {results['median_reward']:.1f}")
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Episode lengths over time
    ax3 = axes[1, 0]
    ax3.plot(lengths, alpha=0.6, linewidth=1)
    ax3.plot(np.convolve(lengths, np.ones(10)/10, mode='valid'), 
             color='red', linewidth=2, label='10-episode MA')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Episode Lengths')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Episodes: {results['num_episodes']}
    
    Mean Reward: {results['mean_reward']:.2f}
    Std Reward: {results['std_reward']:.2f}
    Median Reward: {results['median_reward']:.2f}
    
    Min Reward: {results['min_reward']:.2f}
    Max Reward: {results['max_reward']:.2f}
    
    Mean Length: {results['mean_length']:.1f} steps
    
    Timestamp: {results['timestamp']}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_agents(results_list: List[Dict], save_path: Optional[str] = None):
    """
    Compare multiple agents side-by-side.
    
    Parameters
    ----------
    results_list : list of dict
        List of results dictionaries from evaluate_agent
    save_path : str, optional
        If provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Agent Comparison', fontsize=16, fontweight='bold')
    
    agent_names = [r['agent_name'] for r in results_list]
    mean_rewards = [r['mean_reward'] for r in results_list]
    std_rewards = [r['std_reward'] for r in results_list]
    
    # 1. Bar chart with error bars
    ax1 = axes[0]
    x_pos = np.arange(len(agent_names))
    ax1.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(agent_names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Mean Reward Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Box plot
    ax2 = axes[1]
    reward_data = [r['episode_rewards'] for r in results_list]
    bp = ax2.boxplot(reward_data, labels=agent_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_xticklabels(agent_names, rotation=45, ha='right')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Learning curves (moving average)
    ax3 = axes[2]
    for result in results_list:
        rewards = result['episode_rewards']
        ma = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax3.plot(ma, label=result['agent_name'], linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward (10-ep MA)')
    ax3.set_title('Performance Over Episodes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage - Evaluate all DQN variants + Random Baseline
    from dqn.dqn_initial import DQNAgent
    from dqn.dqn_double import DoubleDQNAgent
    from dqn.dqn_dueling import DuelingDQNAgent
    from dqn.dqn_per_buffer import PerDQNAgent
    from dqn.dqn_noisy import NoisyDQNAgent
    from baseline.random_policy import RandomAgent
    
    # Environment details
    env_id = "ALE/SpaceInvaders-v5"
    input_shape = (4, 84, 84)
    num_actions = 6
    
    # Evaluate all agents
    all_results = []
    
    # Configure all agents to evaluate
    agents_config = [
        ("DQN (Baseline)", DQNAgent, "dqn/models/dqn_initial.pt"),
        ("Double DQN", DoubleDQNAgent, "dqn/models/dqn_double.pt"),
        ("Dueling DQN", DuelingDQNAgent, "dqn/models/dqn_dueling.pt"),
        ("PER DQN", PerDQNAgent, "dqn/models/dqn_per.pt"),
        ("Noisy DQN", NoisyDQNAgent, "dqn/models/dqn_noisy.pt"),
    ]
    
    print("\n" + "="*70)
    print("DQN VARIANTS EVALUATION")
    print("="*70)
    print(f"Environment: {env_id}")
    print(f"Episodes per agent: 100")
    print(f"Seed: 42 (for reproducibility)")
    print("="*70 + "\n")
    
    # First, evaluate Random Baseline (no model loading needed)
    print(f"\n{'='*70}")
    print("Evaluating: Random Baseline")
    print(f"{'='*70}")
    
    try:
        # Create environment to get action space
        from utils.env_utils import make_env
        temp_env = make_env(env_id=env_id, eval_mode=True)
        random_agent = RandomAgent(temp_env.action_space)
        temp_env.close()
        
        print("Created random agent (no model loading required)")
        
        # Evaluate random agent
        results = evaluate_agent(
            agent=random_agent,
            env_id=env_id,
            num_episodes=100,
            seed=42,
            agent_name="random_baseline",
            record_video=True,
            video_freq=20,
            deterministic=False  # Random agent is always stochastic
        )
        
        all_results.append(results)
        
    except Exception as e:
        print(f"!! Error evaluating Random Baseline: {e} !!")
        import traceback
        traceback.print_exc()
    
    # Then evaluate trained agents
    for name, agent_class, model_path in agents_config:
        if os.path.exists(model_path):
            print(f"\n{'='*70}")
            print(f"Evaluating: {name}")
            print(f"{'='*70}")
            
            try:
                # Load agent
                agent = load_agent(agent_class, model_path, input_shape, num_actions)
                
                # Evaluate
                results = evaluate_agent(
                    agent=agent,
                    env_id=env_id,
                    num_episodes=100,
                    seed=42,
                    agent_name=name.lower().replace(" ", "_"),
                    record_video=True,
                    video_freq=20,
                    deterministic=True
                )
                
                all_results.append(results)
                
            except Exception as e:
                print(f"!! Error evaluating {name}: {e} !!")
                import traceback
                traceback.print_exc()
        else:
            print(f"!! Model not found: {model_path} !!")
            print(f"   Skipping {name}")
    
    # Compare all agents
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("CREATING COMPARISON PLOTS")
        print("="*70)
        
        # Sort by mean reward for better visualization
        all_results_sorted = sorted(all_results, key=lambda x: x['mean_reward'], reverse=True)
        
        # Create comparison
        compare_agents(all_results_sorted, save_path="results/agent_comparison.png")
        print("Comparison saved to results/agent_comparison.png")
        
        # Print ranking
        print("\n" + "="*70)
        print("AGENT RANKING (by mean reward)")
        print("="*70)
        for i, result in enumerate(all_results_sorted, 1):
            name = result['agent_name'].replace('_', ' ').title()
            print(f"{i}. {name:25s} | "
                  f"Mean: {result['mean_reward']:7.2f} ± {result['std_reward']:6.2f}")
        print("="*70 + "\n")
        
        # Save summary JSON
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'environment': env_id,
            'num_episodes': 100,
            'seed': 42,
            'agents': [
                {
                    'name': r['agent_name'],
                    'mean_reward': r['mean_reward'],
                    'std_reward': r['std_reward'],
                    'median_reward': r['median_reward'],
                    'min_reward': r['min_reward'],
                    'max_reward': r['max_reward']
                }
                for r in all_results_sorted
            ]
        }
        
        summary_path = "results/evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Summary saved to {summary_path}")
        
    elif len(all_results) == 1:
        print("\n!!  Only one agent evaluated. Need at least 2 for comparison. !!")
    else:
        print("\n!! No agents successfully evaluated. Check model paths. !!")