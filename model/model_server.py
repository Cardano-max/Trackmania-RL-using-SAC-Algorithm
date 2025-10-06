#!/usr/bin/env python3
"""
TrackMania Model Server  
Handles SAC agent, training, and communication with environment container
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import aiohttp
import threading
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from collections import deque
    import random
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using mock model for testing.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    learning_rate_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    batch_size: int = 256
    buffer_size: int = 1000000
    min_buffer_size: int = 1000
    target_entropy: float = -3.0

class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

if TORCH_AVAILABLE:
    class SACNetwork(nn.Module):
        """SAC Actor-Critic Network"""
        
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
            
            # Twin critic networks
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

class SACAgent:
    """Soft Actor-Critic Agent"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.obs_dim = 1 + 4*19 + 2*3  # speed + lidar + previous actions
        self.action_dim = 3  # gas, brake, steering
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if TORCH_AVAILABLE:
            # Networks
            self.network = SACNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.target_network = SACNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.target_network.load_state_dict(self.network.state_dict())
            
            # Optimizers
            self.actor_optimizer = optim.Adam(
                list(self.network.feature_net.parameters()) + 
                list(self.network.actor_mean.parameters()) + 
                list(self.network.actor_log_std.parameters()),
                lr=config.learning_rate_actor
            )
            
            self.critic_optimizer = optim.Adam(
                list(self.network.critic1.parameters()) + 
                list(self.network.critic2.parameters()),
                lr=config.learning_rate_critic
            )
            
            # Automatic entropy tuning
            self.log_alpha = torch.tensor(np.log(config.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate_alpha)
            self.target_entropy = config.target_entropy
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # State tracking
        self.last_obs = None
        self.last_action = None
        self.episode_rewards = []
        self.training_steps = 0
        
        logger.info("SAC Agent initialized")
    
    def preprocess_observation(self, obs_dict: Dict) -> np.ndarray:
        """Convert observation dict to flat array"""
        # Extract components
        speed = [obs_dict["speed"] / 300.0]  # Normalize speed
        lidar = obs_dict["lidar"]  # Already normalized 0-1
        
        # Use last action if available, otherwise zeros
        if self.last_action is not None:
            prev_actions = list(self.last_action) + [0.0, 0.0, 0.0]  # Pad for history
        else:
            prev_actions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Concatenate all features
        flat_obs = np.array(speed + lidar + prev_actions, dtype=np.float32)
        return flat_obs
    
    def get_action(self, observation: Dict, deterministic: bool = False) -> np.ndarray:
        """Get action from observation"""
        obs = self.preprocess_observation(observation)
        
        if not TORCH_AVAILABLE:
            # Simple heuristic policy
            lidar = obs[1:20]  # LIDAR readings
            center_distance = lidar[9]  # Center beam
            
            if center_distance > 0.7:
                return np.array([1.0, 0.0, 0.0])  # Full gas
            elif center_distance > 0.3:
                return np.array([0.5, 0.0, 0.2 if lidar[14] < lidar[4] else -0.2])
            else:
                return np.array([0.0, 0.5, 0.5 if lidar[14] < lidar[4] else -0.5])
        
        # PyTorch-based action selection
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, _, _ = self.network.get_action(obs_tensor, deterministic)
            action = action.cpu().numpy()[0]
        
        # Clip to valid ranges
        action = np.clip(action, -1, 1)
        self.last_action = action.copy()
        
        return action
    
    def store_transition(self, obs: Dict, action: np.ndarray, reward: float, 
                        next_obs: Dict, done: bool) -> None:
        """Store experience in replay buffer"""
        if self.last_obs is not None:
            obs_processed = self.preprocess_observation(self.last_obs)
            next_obs_processed = self.preprocess_observation(obs)
            
            self.replay_buffer.push(
                obs_processed, action, reward, next_obs_processed, done
            )
        
        self.last_obs = obs.copy()
    
    def update(self) -> Dict[str, float]:
        """Update SAC networks"""
        if not TORCH_AVAILABLE or len(self.replay_buffer) < self.config.min_buffer_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return {}
        
        states, actions, rewards, next_states, dones = batch
        
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
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            q_target = rewards + (1 - dones) * self.config.gamma * q_next
        
        q1_current, q2_current = self.network.get_q_values(states, actions)
        critic_loss = F.mse_loss(q1_current, q_target) + F.mse_loss(q2_current, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs, _ = self.network.get_action(states)
        q1_new, q2_new = self.network.get_q_values(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (entropy temperature)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        self.training_steps += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'buffer_size': len(self.replay_buffer)
        }
    
    def save(self, path: str) -> None:
        """Save model"""
        if TORCH_AVAILABLE:
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'training_steps': self.training_steps,
                'config': asdict(self.config)
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model"""
        if TORCH_AVAILABLE and Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.training_steps = checkpoint['training_steps']
            logger.info(f"Model loaded from {path}")

class EnvironmentClient:
    """Client for communicating with environment container"""
    
    def __init__(self, env_url: str = "http://environment:8080"):
        self.env_url = env_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_action(self, agent_id: str, action: np.ndarray) -> Optional[Dict]:
        """Send action to environment and get state"""
        action_data = {
            "agent_id": agent_id,
            "action": {
                "gas": float(action[0]),
                "brake": float(action[1]),
                "steering": float(action[2])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            async with self.session.post(f"{self.env_url}/api/action", json=action_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Environment error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Communication error: {e}")
            return None
    
    async def reset_agent(self, agent_id: str) -> Optional[Dict]:
        """Reset agent in environment"""
        try:
            async with self.session.post(f"{self.env_url}/api/reset/{agent_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Reset error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Reset communication error: {e}")
            return None
    
    async def start_recording(self, race_id: str = None) -> bool:
        """Start race recording"""
        try:
            data = {"race_id": race_id} if race_id else {}
            async with self.session.post(f"{self.env_url}/api/recording/start", json=data) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Recording start error: {e}")
            return False
    
    async def stop_recording(self) -> Optional[str]:
        """Stop race recording"""
        try:
            async with self.session.post(f"{self.env_url}/api/recording/stop") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("race_id")
                return None
        except Exception as e:
            logger.error(f"Recording stop error: {e}")
            return None

# Global instances
config = TrainingConfig()
agent = SACAgent(config)
training_active = False
training_stats = {
    "episodes": 0,
    "total_steps": 0,
    "average_reward": 0.0,
    "best_reward": float('-inf'),
    "recent_rewards": deque(maxlen=100)
}

# FastAPI app
app = FastAPI(title="TrackMania Model Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def training_loop():
    """Main training loop"""
    global training_active, training_stats
    
    agent_id = "sac_agent_1"
    
    async with EnvironmentClient() as env_client:
        logger.info("Starting training loop...")
        
        # Wait for environment to be ready
        await asyncio.sleep(2)
        
        while training_active:
            # Start new episode
            episode_reward = 0.0
            episode_steps = 0
            
            # Reset environment
            state = await env_client.reset_agent(agent_id)
            if state is None:
                await asyncio.sleep(1)
                continue
            
            while training_active and episode_steps < 1000:
                # Get action from agent
                action = agent.get_action(state)
                
                # Send action to environment
                next_state = await env_client.send_action(agent_id, action)
                if next_state is None:
                    break
                
                # Store experience and update
                agent.store_transition(state, action, next_state["reward"], next_state, next_state["done"])
                
                # Update networks
                if len(agent.replay_buffer) >= config.min_buffer_size:
                    update_info = agent.update()
                
                episode_reward += next_state["reward"]
                episode_steps += 1
                training_stats["total_steps"] += 1
                
                state = next_state
                
                if next_state["done"]:
                    break
                
                await asyncio.sleep(0.05)  # 20Hz
            
            # Episode finished
            training_stats["episodes"] += 1
            training_stats["recent_rewards"].append(episode_reward)
            training_stats["average_reward"] = np.mean(training_stats["recent_rewards"])
            training_stats["best_reward"] = max(training_stats["best_reward"], episode_reward)
            
            logger.info(f"Episode {training_stats['episodes']}: Reward={episode_reward:.2f}, Steps={episode_steps}")
            
            # Save model periodically
            if training_stats["episodes"] % 50 == 0:
                agent.save(f"/data/model_episode_{training_stats['episodes']}.pth")

@app.post("/api/training/start")
async def start_training(background_tasks: BackgroundTasks):
    """Start training"""
    global training_active
    
    if training_active:
        return {"status": "already_running"}
    
    training_active = True
    background_tasks.add_task(training_loop)
    logger.info("Training started")
    
    return {"status": "started", "timestamp": datetime.now().isoformat()}

@app.post("/api/training/stop")
async def stop_training():
    """Stop training"""
    global training_active
    training_active = False
    logger.info("Training stopped")
    
    return {"status": "stopped", "timestamp": datetime.now().isoformat()}

@app.get("/api/training/status")
async def get_training_status():
    """Get training status"""
    return {
        "active": training_active,
        "stats": training_stats,
        "config": asdict(config),
        "pytorch_available": TORCH_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/model/save")
async def save_model():
    """Save current model"""
    path = f"/data/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    agent.save(path)
    return {"status": "saved", "path": path}

@app.post("/api/model/load")
async def load_model(path: str):
    """Load model from path"""
    try:
        agent.load(path)
        return {"status": "loaded", "path": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/action/single")
async def get_single_action(observation: dict):
    """Get single action from observation (for testing)"""
    action = agent.get_action(observation, deterministic=True)
    return {
        "action": {
            "gas": float(action[0]),
            "brake": float(action[1]), 
            "steering": float(action[2])
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/status")
async def get_status():
    """Get model server status"""
    return {
        "status": "running",
        "training_active": training_active,
        "pytorch_available": TORCH_AVAILABLE,
        "device": str(agent.device) if TORCH_AVAILABLE else "cpu",
        "buffer_size": len(agent.replay_buffer),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with info"""
    return HTMLResponse("""
    <html>
        <head><title>TrackMania Model Server</title></head>
        <body>
            <h1>TrackMania Model Server</h1>
            <p>Model container is running!</p>
            <h2>Features:</h2>
            <ul>
                <li>SAC (Soft Actor-Critic) Agent</li>
                <li>Experience Replay Buffer</li>
                <li>Automatic Training Loop</li>
                <li>Model Save/Load</li>
            </ul>
            <h2>API Endpoints:</h2>
            <ul>
                <li><b>POST /api/training/start</b> - Start training</li>
                <li><b>POST /api/training/stop</b> - Stop training</li>
                <li><b>GET /api/training/status</b> - Training status</li>
                <li><b>POST /api/model/save</b> - Save model</li>
                <li><b>POST /api/model/load</b> - Load model</li>
                <li><b>POST /api/action/single</b> - Get action (testing)</li>
                <li><b>GET /api/status</b> - Server status</li>
            </ul>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    logger.info("Starting TrackMania Model Server...")
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")