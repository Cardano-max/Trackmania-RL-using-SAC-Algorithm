#!/usr/bin/env python3
"""
Standalone SAC training script for TrackMania RL
Can run with or without the full TMRL framework
"""

import os
import sys
import json
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using mock trainer for testing.")
from collections import deque
import random
from pathlib import Path
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional
import logging
import time

# Import our mock environment
sys.path.append('/app/scripts')
from setup_env import create_mock_environment


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class SACNetwork(nn.Module):
    """SAC Actor-Critic Network for LIDAR observations"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (Policy) network
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic networks (Q-functions)
        self.critic1 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.feature_net(state)
        return features
    
    def get_action(self, state, deterministic=False):
        features = self.forward(state)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            return action, None, None
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)
    
    def get_q_values(self, state, action):
        features = self.forward(state)
        q_input = torch.cat([features, action], dim=1)
        q1 = self.critic1(q_input)
        q2 = self.critic2(q_input)
        return q1, q2


class SACTrainer:
    """Soft Actor-Critic trainer"""
    
    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        
        # Observation preprocessing
        self.obs_dim = self._get_obs_dim()
        self.action_dim = env.action_space.shape[0]
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = SACNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_network = SACNetwork(self.obs_dim, self.action_dim).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.network.feature_net.parameters()) + 
            list(self.network.actor_mean.parameters()) + 
            list(self.network.actor_log_std.parameters()),
            lr=config.get('lr_actor', 3e-4)
        )
        
        self.critic_optimizer = optim.Adam(
            list(self.network.critic1.parameters()) + 
            list(self.network.critic2.parameters()),
            lr=config.get('lr_critic', 3e-4)
        )
        
        # SAC parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2)
        self.target_entropy = -self.action_dim
        
        # Automatic entropy tuning
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.get('lr_alpha', 3e-4))
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.get('buffer_size', 1000000))
        
        # Training parameters
        self.batch_size = config.get('batch_size', 256)
        self.min_buffer_size = config.get('min_buffer_size', 1000)
        
        # Logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path("/TmrlData/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_obs_dim(self):
        """Calculate observation dimension"""
        # Speed (1) + LIDAR (4*19) + Previous actions (2*3)
        return 1 + 4*19 + 2*3
    
    def _preprocess_obs(self, obs):
        """Preprocess observation tuple into flat vector"""
        speed, lidar, prev_action1, prev_action2 = obs
        
        # Flatten and concatenate
        flat_obs = np.concatenate([
            speed.flatten(),
            lidar.flatten(),
            prev_action1.flatten(),
            prev_action2.flatten()
        ])
        
        return flat_obs.astype(np.float32)
    
    def update(self):
        """Update SAC networks"""
        if len(self.replay_buffer) < self.min_buffer_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.network.get_action(next_states)
            q1_next, q2_next = self.target_network.get_q_values(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1_current, q2_current = self.network.get_q_values(states, actions)
        critic_loss = F.mse_loss(q1_current, q_target) + F.mse_loss(q2_current, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs, _ = self.network.get_action(states)
        q1_new, q2_new = self.network.get_q_values(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature parameter)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def train(self, num_episodes: int = 1000):
        """Main training loop"""
        self.logger.info(f"Starting SAC training for {num_episodes} episodes")
        
        episode_rewards = []
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs = self._preprocess_obs(obs)
            
            episode_reward = 0
            episode_steps = 0
            
            while True:
                # Select action
                if len(self.replay_buffer) < self.min_buffer_size:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        action, _, _ = self.network.get_action(obs_tensor)
                        action = action.cpu().numpy()[0]
                
                # Take step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_obs = self._preprocess_obs(next_obs)
                
                # Store transition
                done = terminated or truncated
                self.replay_buffer.push(obs, action, reward, next_obs, done)
                
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                
                # Update networks
                if len(self.replay_buffer) >= self.min_buffer_size:
                    update_info = self.update()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                self.logger.info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Avg(10)={avg_reward:.2f}, Steps={episode_steps}, "
                    f"Buffer={len(self.replay_buffer)}"
                )
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_model("best_model.pth")
            
            # Save checkpoint
            if episode % 100 == 0 and episode > 0:
                self.save_model(f"checkpoint_{episode}.pth")
        
        self.logger.info("Training completed!")
        self.save_model("final_model.pth")
    
    def save_model(self, filename: str):
        """Save model weights"""
        weights_dir = Path("/TmrlData/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
        }, weights_dir / filename)
        
        self.logger.info(f"Model saved: {filename}")


def load_config():
    """Load training configuration"""
    config_path = Path("/TmrlData/config/config.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('ALG', {})
    
    # Default configuration
    return {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'lr_alpha': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'batch_size': 256,
        'buffer_size': 1000000,
        'min_buffer_size': 1000
    }


def main():
    """Main training function"""
    print("Starting standalone SAC training...")
    
    # Create environment
    env = create_mock_environment()
    
    # Load configuration
    config = load_config()
    
    # Create trainer
    trainer = SACTrainer(env, config)
    
    # Start training
    trainer.train(num_episodes=1000)


if __name__ == "__main__":
    main()