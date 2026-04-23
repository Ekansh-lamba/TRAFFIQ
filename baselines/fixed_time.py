import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from env.traffic_env import TrafficEnv


class FixedTimeController:
    """
    Fixed-time traffic signal controller.
    Cycles through phases 0→1→2→3→0 every 30 seconds regardless of traffic conditions.
    """
    
    def __init__(self, env, phase_duration=30):
        """
        Initialize fixed-time controller.
        
        Args:
            env (TrafficEnv): Traffic environment instance
            phase_duration (int): Duration of each phase in decision steps (default: 30)
        """
        self.env = env
        self.phase_duration = phase_duration
        self.num_phases = 4
        self.current_phase_idx = 0
        self.steps_in_current_phase = 0
    
    def get_action(self):
        """
        Get the next action based on fixed-time schedule.
        
        Returns:
            int: Phase index (0-3)
        """
        # Check if it's time to switch to next phase
        if self.steps_in_current_phase >= self.phase_duration:
            # Move to next phase
            self.current_phase_idx = (self.current_phase_idx + 1) % self.num_phases
            self.steps_in_current_phase = 0
        
        self.steps_in_current_phase += 1
        return self.current_phase_idx
    
    def reset(self):
        """Reset the controller state for a new episode."""
        self.current_phase_idx = 0
        self.steps_in_current_phase = 0


def run_fixed_time_baseline(num_episodes=20, phase_duration=30):
    """
    Run fixed-time baseline evaluation.
    
    Args:
        num_episodes (int): Number of episodes to run (default: 20)
        phase_duration (int): Duration of each phase in decision steps (default: 30)
    
    Returns:
        dict: Results containing per-episode metrics and summary statistics
    """
    # Initialize environment in headless mode
    env = TrafficEnv(
        sumo_config=r"sumo_files\intersection.sumocfg",
        gui=False,
        port=8813,
        max_steps=1000
    )
    
    # Initialize fixed-time controller
    controller = FixedTimeController(env, phase_duration=phase_duration)
    
    # Storage for results
    episode_results = []
    
    print("="*60)
    print("FIXED-TIME BASELINE EVALUATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Phase Duration: {phase_duration} decision steps")
    print(f"  Max Steps per Episode: {env.max_steps}")
    print("="*60)
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Reset environment and controller
        state = env.reset()
        controller.reset()
        
        episode_reward = 0
        done = False
        step = 0
        
        # Run episode
        while not done:
            # Get action from fixed-time controller
            action = controller.get_action()
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Print progress every 200 steps
            if step % 200 == 0:
                print(f"  Step {step}: Queue={info['queue_length']:.1f}, "
                      f"Waiting={info['waiting_time']:.1f}s, Arrived={info['arrived']}")
        
        # Get episode metrics
        metrics = env.get_metrics()
        metrics['total_reward'] = episode_reward
        episode_results.append(metrics)
        
        # Print episode summary
        print(f"\n  Episode Summary:")
        print(f"    Avg Waiting Time: {metrics['avg_waiting_time']:.2f}s")
        print(f"    Avg Queue Length: {metrics['avg_queue_length']:.2f}")
        print(f"    Throughput: {metrics['throughput']} vehicles")
        print(f"    Total Reward: {metrics['total_reward']:.2f}")
    
    # Close environment
    env.close()
    
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
    print("FINAL RESULTS (20 episodes)")
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
            'num_episodes': num_episodes,
            'phase_duration': phase_duration,
            'max_steps_per_episode': 1000
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
    # Run fixed-time baseline
    results = run_fixed_time_baseline(num_episodes=20, phase_duration=30)
    
    # Save results
    save_results(results, r"results\fixed_time_results.json")
    
    print("\n✅ Fixed-time baseline evaluation complete!")