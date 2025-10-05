#!/usr/bin/env python3
"""
Simple test script to verify training pipeline works without PyTorch
"""

import numpy as np
import sys
from setup_env import create_mock_environment

class MockTrainer:
    """Mock trainer for testing without PyTorch"""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.episode_count = 0
        
    def select_action(self, obs):
        """Simple heuristic policy for testing"""
        speed, lidar, _, _ = obs
        
        # Simple logic: accelerate if clear ahead, steer if obstacles
        center_distance = lidar[0, 9]  # Center beam of latest LIDAR
        
        if center_distance > 0.7:  # Clear ahead
            return np.array([1.0, 0.0, 0.0])  # Full acceleration
        elif center_distance > 0.3:  # Some obstacle
            return np.array([0.5, 0.0, 0.2])  # Moderate speed, slight turn
        else:  # Close obstacle
            return np.array([0.0, 0.5, 0.5])  # Brake and turn
    
    def train_episode(self):
        """Train for one episode"""
        obs, info = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = self.select_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        self.episode_count += 1
        return total_reward, steps, info
    
    def train(self, num_episodes=10):
        """Train for multiple episodes"""
        print(f"Starting mock training for {num_episodes} episodes...")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            reward, steps, info = self.train_episode()
            episode_rewards.append(reward)
            
            print(f"Episode {episode + 1}: "
                  f"Reward={reward:.2f}, "
                  f"Steps={steps}, "
                  f"Completion={info.get('track_completion', 0):.3f}")
        
        avg_reward = np.mean(episode_rewards)
        print(f"\nTraining completed!")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Best episode: {max(episode_rewards):.2f}")
        print(f"Episodes completed: {num_episodes}")
        
        return episode_rewards

def test_training_pipeline():
    """Test the complete training pipeline"""
    print("=== Testing Training Pipeline ===\n")
    
    # Create environment
    env = create_mock_environment()
    
    # Mock config
    config = {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'alpha': 0.2
    }
    
    # Create trainer
    trainer = MockTrainer(env, config)
    
    # Run training
    rewards = trainer.train(num_episodes=5)
    
    # Verify results
    assert len(rewards) == 5, "Should complete 5 episodes"
    assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
    
    print("✅ Training pipeline test passed!")
    return True

def test_action_selection():
    """Test action selection logic"""
    print("\n=== Testing Action Selection ===\n")
    
    env = create_mock_environment()
    trainer = MockTrainer(env, {})
    
    # Test different scenarios
    obs, _ = env.reset()
    
    # Test several actions
    for i in range(3):
        action = trainer.select_action(obs)
        print(f"Action {i+1}: gas={action[0]:.2f}, brake={action[1]:.2f}, steer={action[2]:.2f}")
        
        # Verify action is valid
        assert len(action) == 3, "Action should have 3 components"
        assert all(-1 <= a <= 1 for a in action), "Actions should be in [-1, 1]"
        
        # Step environment
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    print("✅ Action selection test passed!")
    return True

def main():
    """Run all tests"""
    print("=== TrackMania RL Training Test Suite ===\n")
    
    tests = [
        test_action_selection,
        test_training_pipeline,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== Test Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("🎉 All training tests passed! Ready for Docker deployment.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()