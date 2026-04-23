from env.traffic_env import TrafficEnv
import time

def test_environment():
    """Test the traffic environment with visualization."""
    
    print("Initializing Traffic Environment...")
    env = TrafficEnv(
        sumo_config=r"sumo_files\intersection.sumocfg",
        gui=True,  # Use GUI for visualization
        port=8813,
        max_steps=100  # Short test episode
    )
    
    print("Resetting environment...")
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state values: {state}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")
    
    print("\nRunning 100 steps with random actions...")
    total_reward = 0
    
    for step in range(100):
        # Take random action
        action = env.action_space.sample()
        
        # Execute step
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            print(f"Step {step+1}: action={action}, reward={reward:.2f}, "
                  f"waiting_time={info['waiting_time']:.1f}, queue={info['queue_length']:.1f}")
        
        if done:
            print(f"\nEpisode finished at step {step+1}")
            break
    
    # Get final metrics
    metrics = env.get_metrics()
    print(f"\n{'='*50}")
    print("Episode Metrics:")
    print(f"  Average Waiting Time: {metrics['avg_waiting_time']:.2f}")
    print(f"  Average Queue Length: {metrics['avg_queue_length']:.2f}")
    print(f"  Throughput (vehicles): {metrics['throughput']}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"{'='*50}")
    
    env.close()
    print("\n✅ Environment test complete!")

if __name__ == "__main__":
    test_environment()