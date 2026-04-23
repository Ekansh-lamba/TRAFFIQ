import numpy as np
from env.traffic_env import TrafficEnv
from env import env_utils

# Start environment to get TraCI connection
env = TrafficEnv(sumo_config=r"sumo_files\intersection.sumocfg", gui=False, port=8813)
state = env.reset()

print("Testing env_utils functions...")
print("="*50)

# Test normalize function
print("\n1. Testing normalize():")
print(f"   normalize(10, 20) = {env_utils.normalize(10, 20)}")
print(f"   normalize(25, 20) = {env_utils.normalize(25, 20)} (should clamp to 1.0)")
print(f"   normalize(5, 0) = {env_utils.normalize(5, 0)} (should return 0.0)")
assert env_utils.normalize(10, 20) == 0.5, "normalize(10, 20) should be 0.5"
assert env_utils.normalize(25, 20) == 1.0, "normalize(25, 20) should clamp to 1.0"
assert env_utils.normalize(5, 0) == 0.0, "normalize(5, 0) should return 0.0"
print("   ✓ normalize() works correctly")

# Test get_lane_queue
print("\n2. Testing get_lane_queue():")
for lane in env.lane_ids:
    queue = env_utils.get_lane_queue(lane)
    print(f"   Lane {lane}: queue = {queue}")
    assert isinstance(queue, int), f"Queue should be int, got {type(queue)}"
print("   ✓ get_lane_queue() works correctly")

# Test get_lane_waiting_time
print("\n3. Testing get_lane_waiting_time():")
for lane in env.lane_ids:
    waiting_time = env_utils.get_lane_waiting_time(lane)
    print(f"   Lane {lane}: waiting_time = {waiting_time:.2f}s")
    assert isinstance(waiting_time, (int, float)), f"Waiting time should be numeric, got {type(waiting_time)}"
print("   ✓ get_lane_waiting_time() works correctly")

# Test get_state
print("\n4. Testing get_state():")
state_from_utils = env_utils.get_state(
    tls_id=env.tls_id,
    lane_ids=env.lane_ids,
    max_vehicles=env.max_vehicles,
    num_phases=env.num_phases,
    max_phase_time=env.max_phase_time,
    steps_in_phase=env.steps_since_last_switch
)
print(f"   State shape: {state_from_utils.shape}")
print(f"   State values: {state_from_utils}")
print(f"   State dtype: {state_from_utils.dtype}")
assert state_from_utils.shape == (6,), f"State should have shape (6,), got {state_from_utils.shape}"
assert state_from_utils.dtype == np.float32, f"State dtype should be float32, got {state_from_utils.dtype}"
assert np.all((state_from_utils >= 0) & (state_from_utils <= 1)), "All state values should be in [0, 1]"
print("   ✓ get_state() works correctly")

# Test compute_reward
print("\n5. Testing compute_reward():")
reward = env_utils.compute_reward(env.lane_ids)
print(f"   Reward: {reward:.2f}")
assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
assert reward <= 0, "Reward should be negative or zero"
print("   ✓ compute_reward() works correctly")

# Test get_metrics
print("\n6. Testing get_metrics():")
metrics = env_utils.get_metrics(env.lane_ids, env.total_arrived)
print(f"   Metrics: {metrics}")
assert 'avg_waiting_time' in metrics, "Missing avg_waiting_time"
assert 'avg_queue_length' in metrics, "Missing avg_queue_length"
assert 'throughput' in metrics, "Missing throughput"
assert isinstance(metrics['avg_waiting_time'], float), "avg_waiting_time should be float"
assert isinstance(metrics['avg_queue_length'], float), "avg_queue_length should be float"
assert isinstance(metrics['throughput'], int), "throughput should be int"
print("   ✓ get_metrics() works correctly")

env.close()

print("\n" + "="*50)
print("✅ STEP 3 VERIFICATION COMPLETE!")
print("="*50)
print("\nAll utility functions working correctly:")
print("  ✓ normalize() - clamps and normalizes to [0,1]")
print("  ✓ get_lane_queue() - returns halting vehicle count")
print("  ✓ get_lane_waiting_time() - returns waiting time")
print("  ✓ get_state() - returns normalized 6-element float32 array")
print("  ✓ compute_reward() - returns negative (waiting_time + queue)")
print("  ✓ get_metrics() - returns dict with all required keys")
print("\nReady to proceed to Step 4!")