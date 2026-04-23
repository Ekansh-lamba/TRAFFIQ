import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.traffic_env import TrafficEnv


class TrafficEnvWrapper(gym.Env):
    """
    Gym wrapper for TrafficEnv to make it compatible with Stable-Baselines3.
    """
    
    def __init__(self, sumo_config, gui=False, port=8813, max_steps=1000):
        """
        Initialize the wrapped environment.
        
        Args:
            sumo_config (str): Path to SUMO configuration file
            gui (bool): If True, use sumo-gui.exe; if False, use sumo.exe
            port (int): TraCI port for connection
            max_steps (int): Maximum number of decision steps per episode
        """
        super(TrafficEnvWrapper, self).__init__()
        
        # Initialize the base environment
        self.env = TrafficEnv(sumo_config, gui, port, max_steps)
        
        # Set observation and action spaces from base environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        
        Returns:
            tuple: (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        obs = self.env.reset()
        info = {}
        return obs, info
    
    def step(self, action):
        """
        Execute one step.
        
        Args:
            action: Action to take
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.env.step(action)
        
        # Stable-Baselines3 uses separate terminated and truncated flags
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_metrics(self):
        """Get episode metrics from base environment."""
        return self.env.get_metrics()


def make_env(sumo_config, gui=False, port=8813, max_steps=1000):
    """
    Create and wrap the environment.
    
    Args:
        sumo_config (str): Path to SUMO configuration file
        gui (bool): Whether to use GUI
        port (int): TraCI port
        max_steps (int): Max steps per episode
    
    Returns:
        Monitor: Wrapped environment
    """
    env = TrafficEnvWrapper(sumo_config, gui, port, max_steps)
    env = Monitor(env)
    return env


def train_ppo(total_timesteps=200000, eval_episodes=20):
    """
    Train PPO agent on traffic environment.
    
    Args:
        total_timesteps (int): Total training timesteps (default: 200,000)
        eval_episodes (int): Number of evaluation episodes (default: 20)
    
    Returns:
        dict: Training and evaluation results
    """
    print("="*60)
    print("PPO TRAINING - CONDITION B (Ground Truth TraCI State)")
    print("="*60)
    
    # Create directories
    os.makedirs("results/tensorboard_logs", exist_ok=True)
    os.makedirs("results/best_ppo_groundtruth", exist_ok=True)
    
    # Create training environment
    print("\nInitializing training environment...")
    train_env = make_env(
        sumo_config=r"sumo_files\intersection.sumocfg",
        gui=False,
        port=8813,
        max_steps=1000
    )
    
    # Create evaluation environment (different port to avoid conflicts)
    print("Initializing evaluation environment...")
    eval_env = make_env(
        sumo_config=r"sumo_files\intersection.sumocfg",
        gui=False,
        port=8814,
        max_steps=1000
    )
    
    # Create PPO model
    print("\nCreating PPO model...")
    print("Hyperparameters:")
    print("  Policy: MlpPolicy")
    print("  Learning Rate: 3e-4")
    print("  n_steps: 2048")
    print("  batch_size: 64")
    print("  n_epochs: 10")
    print("  gamma: 0.99")
    print("  clip_range: 0.2")
    
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="results/tensorboard_logs/"
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="results/best_ppo_groundtruth/",
        log_path="results/best_ppo_groundtruth/",
        eval_freq=10000,  # Evaluate every 10,000 steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="results/checkpoints_ppo_groundtruth/",
        name_prefix="ppo_model"
    )
    
    # Train the model
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("="*60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Save final model
    model_path = "results/ppo_groundtruth_model.zip"
    model.save(model_path)
    print(f"\n✓ Final model saved to: {model_path}")
    
    # Close training environment
    train_env.close()
    
    # Run final evaluation
    print(f"\nRunning final evaluation ({eval_episodes} episodes)...")
    print("-" * 60)
    
    episode_results = []
    
    for episode in range(eval_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        # Get metrics
        metrics = eval_env.unwrapped.get_metrics()
        metrics['total_reward'] = float(episode_reward)
        episode_results.append(metrics)
        
        print(f"Episode {episode + 1}/{eval_episodes}: "
              f"Waiting={metrics['avg_waiting_time']:.2f}s, "
              f"Queue={metrics['avg_queue_length']:.2f}, "
              f"Throughput={metrics['throughput']}, "
              f"Reward={episode_reward:.2f}")
    
    # Close evaluation environment
    eval_env.close()
    
    # Compute summary statistics
    waiting_times = [ep['avg_waiting_time'] for ep in episode_results]
    queue_lengths = [ep['avg_queue_length'] for ep in episode_results]
    throughputs = [ep['throughput'] for ep in episode_results]
    total_rewards = [ep['total_reward'] for ep in episode_results]
    
    summary = {
        'avg_waiting_time': {
            'mean': float(np.mean(waiting_times)),
            'std': float(np.std(waiting_times)),
            'min': float(np.min(waiting_times)),
            'max': float(np.max(waiting_times))
        },
        'avg_queue_length': {
            'mean': float(np.mean(queue_lengths)),
            'std': float(np.std(queue_lengths)),
            'min': float(np.min(queue_lengths)),
            'max': float(np.max(queue_lengths))
        },
        'throughput': {
            'mean': float(np.mean(throughputs)),
            'std': float(np.std(throughputs)),
            'min': int(np.min(throughputs)),
            'max': int(np.max(throughputs))
        },
        'total_reward': {
            'mean': float(np.mean(total_rewards)),
            'std': float(np.std(total_rewards)),
            'min': float(np.min(total_rewards)),
            'max': float(np.max(total_rewards))
        }
    }
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"\nAverage Waiting Time:")
    print(f"  Mean: {summary['avg_waiting_time']['mean']:.2f}s ± {summary['avg_waiting_time']['std']:.2f}s")
    print(f"  Range: [{summary['avg_waiting_time']['min']:.2f}, {summary['avg_waiting_time']['max']:.2f}]")
    
    print(f"\nAverage Queue Length:")
    print(f"  Mean: {summary['avg_queue_length']['mean']:.2f} ± {summary['avg_queue_length']['std']:.2f}")
    print(f"  Range: [{summary['avg_queue_length']['min']:.2f}, {summary['avg_queue_length']['max']:.2f}]")
    
    print(f"\nThroughput:")
    print(f"  Mean: {summary['throughput']['mean']:.0f} ± {summary['throughput']['std']:.2f} vehicles")
    print(f"  Range: [{summary['throughput']['min']}, {summary['throughput']['max']}]")
    
    print(f"\nTotal Reward:")
    print(f"  Mean: {summary['total_reward']['mean']:.2f} ± {summary['total_reward']['std']:.2f}")
    print(f"  Range: [{summary['total_reward']['min']:.2f}, {summary['total_reward']['max']:.2f}]")
    print("="*60)
    
    # Prepare results for saving
    results = {
        'configuration': {
            'algorithm': 'PPO',
            'total_timesteps': total_timesteps,
            'eval_episodes': eval_episodes,
            'hyperparameters': {
                'policy': 'MlpPolicy',
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'clip_range': 0.2
            }
        },
        'episode_results': episode_results,
        'summary': summary
    }
    
    return results


def save_results(results, output_path):
    """
    Save results to JSON file.
    
    Args:
        results (dict): Results dictionary
        output_path (str): Path to save JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    # Train PPO agent
    results = train_ppo(total_timesteps=200000, eval_episodes=20)
    
    # Save results
    save_results(results, r"results\ppo_groundtruth_results.json")
    
    print("\n✅ PPO training and evaluation complete!")
    print("\nTo view training progress in TensorBoard, run:")
    print("  tensorboard --logdir=results/tensorboard_logs/")