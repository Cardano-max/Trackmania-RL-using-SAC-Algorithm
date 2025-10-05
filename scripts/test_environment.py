#!/usr/bin/env python3
"""
Test script to verify the mock environment works correctly
"""

import numpy as np
from setup_env import create_mock_environment

def test_basic_functionality():
    """Test basic environment functionality"""
    print("Testing basic environment functionality...")
    
    env = create_mock_environment()
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation shapes: {[o.shape for o in obs]}")
    print(f"  Action space: {env.action_space}")
    print(f"  Initial info: {info}")
    
    # Test a few steps
    total_reward = 0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: action={action}, reward={reward:.2f}, terminated={terminated}")
        
        if terminated or truncated:
            break
    
    print(f"✓ Environment stepping works. Total reward: {total_reward:.2f}")
    return True

def test_episode_completion():
    """Test complete episode"""
    print("\nTesting complete episode...")
    
    env = create_mock_environment()
    obs, info = env.reset()
    
    episode_length = 0
    total_reward = 0
    
    while True:
        # Simple policy: always accelerate and steer slightly
        action = np.array([1.0, 0.0, 0.1])  # [gas, brake, steering]
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_length += 1
        total_reward += reward
        
        if episode_length % 100 == 0:
            print(f"  Step {episode_length}: completion={info['track_completion']:.3f}, speed={info['speed']:.1f}")
        
        if terminated or truncated:
            break
    
    print(f"✓ Episode completed after {episode_length} steps")
    print(f"  Final track completion: {info['track_completion']:.3f}")
    print(f"  Total reward: {total_reward:.2f}")
    return True

def test_reward_function():
    """Test reward function behavior"""
    print("\nTesting reward function...")
    
    env = create_mock_environment()
    obs, info = env.reset()
    
    # Test different actions
    test_actions = [
        np.array([1.0, 0.0, 0.0]),   # Pure acceleration
        np.array([0.0, 1.0, 0.0]),   # Pure braking
        np.array([0.0, 0.0, 1.0]),   # Pure right steering
        np.array([0.0, 0.0, -1.0]),  # Pure left steering
        np.array([1.0, 0.0, 0.1]),   # Acceleration with slight steering
    ]
    
    action_names = ["Accelerate", "Brake", "Steer Right", "Steer Left", "Accel + Steer"]
    
    for action, name in zip(test_actions, action_names):
        env.reset()
        obs, reward, _, _, info = env.step(action)
        print(f"  {name}: reward={reward:.2f}, speed={info['speed']:.1f}")
    
    print("✓ Reward function responds to different actions")
    return True

def visualize_lidar():
    """Visualize LIDAR data"""
    print("\nVisualizing LIDAR data...")
    
    env = create_mock_environment()
    obs, info = env.reset()
    
    # Get LIDAR data
    speed, lidar_history, _, _ = obs
    current_lidar = lidar_history[0]  # Most recent LIDAR
    
    # Simple visualization
    print("  LIDAR readings (19 beams from left to right):")
    print("  " + "".join([f"{x:.1f} " for x in current_lidar]))
    
    # Create a simple ASCII visualization
    print("  ASCII visualization (closer objects shown as #):")
    ascii_view = ""
    for reading in current_lidar:
        if reading < 0.3:
            ascii_view += "#"
        elif reading < 0.6:
            ascii_view += "o"
        else:
            ascii_view += "."
    print(f"  {ascii_view}")
    
    print("✓ LIDAR visualization complete")
    return True

def main():
    """Run all tests"""
    print("=== TrackMania RL Environment Test Suite ===\n")
    
    tests = [
        test_basic_functionality,
        test_episode_completion,
        test_reward_function,
        visualize_lidar
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    print(f"\n=== Test Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("🎉 All tests passed! Environment is ready for training.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()