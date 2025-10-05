#!/usr/bin/env python3
"""
Environment setup script for TMRL
Creates a mock TrackMania environment for testing without the actual game
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import random
from typing import Tuple, Dict, Any


class MockTrackManiaEnv(gym.Env):
    """Mock TrackMania environment for testing TMRL without the actual game"""
    
    def __init__(self):
        super().__init__()
        
        # LIDAR observations: (speed, 4 last LIDARs, 2 previous actions)
        # Speed: 1D, LIDAR: 4x19, Actions: 2x3
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=300, shape=(1,), dtype=np.float32),  # speed
            spaces.Box(low=0, high=1, shape=(4, 19), dtype=np.float32),  # LIDAR history
            spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),  # previous action
            spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),  # previous action 2
        ))
        
        # Actions: [gas, brake, steering] between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # State variables
        self.speed = 0.0
        self.position = 0.0
        self.track_completion = 0.0
        self.lidar_history = np.zeros((4, 19), dtype=np.float32)
        self.action_history = [np.zeros(3), np.zeros(3)]
        self.step_count = 0
        self.max_steps = 1000
        
        # Track parameters
        self.track_length = 100.0
        self.track_width = 10.0
        
    def reset(self, seed=None, options=None) -> Tuple[Any, Dict]:
        super().reset(seed=seed)
        
        self.speed = 0.0
        self.position = 0.0
        self.track_completion = 0.0
        self.lidar_history = np.random.uniform(0.1, 1.0, (4, 19)).astype(np.float32)
        self.action_history = [np.zeros(3), np.zeros(3)]
        self.step_count = 0
        
        obs = self._get_observation()
        info = {"track_completion": self.track_completion}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        action = np.clip(action, -1, 1)
        
        # Simulate car dynamics
        gas, brake, steering = action
        
        # Update speed based on gas/brake
        acceleration = gas * 2.0 - brake * 3.0
        self.speed = max(0, min(200, self.speed + acceleration))
        
        # Update position based on speed
        self.position += self.speed * 0.01
        self.track_completion = min(1.0, self.position / self.track_length)
        
        # Simulate LIDAR measurements
        self._update_lidar(steering)
        
        # Update action history
        self.action_history[1] = self.action_history[0].copy()
        self.action_history[0] = action.copy()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        self.step_count += 1
        terminated = self.track_completion >= 1.0
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_observation()
        info = {
            "track_completion": self.track_completion,
            "speed": self.speed,
            "position": self.position
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Tuple:
        speed_obs = np.array([self.speed / 200.0], dtype=np.float32)
        return (speed_obs, self.lidar_history, self.action_history[0], self.action_history[1])
    
    def _update_lidar(self, steering: float):
        # Simulate LIDAR measurements based on steering
        # Roll history
        self.lidar_history[1:] = self.lidar_history[:-1]
        
        # Generate new LIDAR measurement
        center = 9  # Center beam
        new_lidar = np.ones(19, dtype=np.float32)
        
        # Add some track boundaries
        for i in range(19):
            beam_angle = (i - center) * 0.1  # Beam angles
            effective_angle = beam_angle + steering * 0.5
            
            # Distance to track boundary
            if abs(effective_angle) > 0.3:  # Off track
                new_lidar[i] = 0.1 + random.uniform(0, 0.2)
            else:  # On track
                new_lidar[i] = 0.8 + random.uniform(0, 0.2)
        
        self.lidar_history[0] = new_lidar
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        reward = 0.0
        
        # Progress reward
        progress_reward = self.track_completion * 100.0
        
        # Speed reward (encourage optimal speed)
        speed_reward = min(self.speed / 100.0, 1.0) * 10.0
        
        # Smooth driving reward (penalize excessive steering)
        steering_penalty = -abs(action[2]) * 2.0
        
        # Stay on track reward (based on center LIDAR readings)
        center_distance = self.lidar_history[0, 9]  # Center beam
        track_reward = center_distance * 5.0
        
        reward = progress_reward + speed_reward + steering_penalty + track_reward
        
        return reward


def create_mock_environment():
    """Create and return a mock TrackMania environment"""
    return MockTrackManiaEnv()


if __name__ == "__main__":
    # Test the mock environment
    env = create_mock_environment()
    
    obs, info = env.reset()
    print(f"Initial observation shape: {[o.shape for o in obs]}")
    print(f"Action space: {env.action_space}")
    
    # Run a few test steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, speed={info['speed']:.1f}, completion={info['track_completion']:.3f}")
        
        if terminated or truncated:
            print("Episode ended!")
            break