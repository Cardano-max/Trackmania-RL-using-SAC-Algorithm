#!/usr/bin/env python3
"""
Synchronized RL Learning + 3D Visualization System
Shows real Q-Learning metrics connected to 3D racing behavior
"""

import asyncio
import json
import time
import math
import random
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLearningAgent:
    """Real Q-Learning Agent with detailed metrics"""
    
    def __init__(self, agent_id: str, name: str, color: str, learning_rate: float = 0.1):
        self.agent_id = agent_id
        self.name = name
        self.color = color
        
        # Q-Learning parameters
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.min_exploration = 0.05
        self.exploration_decay = 0.995
        
        # Learning metrics
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.q_values_history = deque(maxlen=200)
        self.learning_curve = deque(maxlen=50)
        self.policy_losses = deque(maxlen=100)
        self.value_estimates = deque(maxlen=100)
        self.training_time = 0.0
        
        # Racing performance
        self.current_reward = 0.0
        self.best_lap_time = float('inf')
        self.current_lap_time = 0.0
        self.races_won = 0
        self.total_races = 0
        
        # 3D position and behavior
        self.track_position = random.uniform(0, 0.05)
        self.speed = 0.0
        self.lateral_offset = 0.0
        self.actions_taken = {"throttle": 0.0, "brake": 0.0, "steering": 0.0}
        
    def get_state(self, position: float, speed: float, competitors_nearby: int):
        """Discretize state for Q-learning"""
        pos_bucket = int(position * 20) % 20
        speed_bucket = min(4, int(speed / 50))
        comp_bucket = min(2, competitors_nearby)
        return (pos_bucket, speed_bucket, comp_bucket)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy with real Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]  # [throttle, brake, steering]
        
        if random.random() < self.exploration_rate:
            # Exploration: Random action (shows as erratic behavior in 3D)
            throttle = random.uniform(0.2, 1.0)
            brake = random.uniform(0.0, 0.4)
            steering = random.uniform(-1.0, 1.0)
            self.learning_decision = "exploring"
        else:
            # Exploitation: Use learned Q-values (shows as smart behavior in 3D)
            q_values = self.q_table[state]
            self.q_values_history.append(max(q_values))
            
            # Convert Q-values to racing actions
            throttle = max(0.2, min(1.0, 0.7 + q_values[0] * 0.3))
            brake = max(0.0, min(0.5, q_values[1]))
            steering = max(-1.0, min(1.0, q_values[2]))
            self.learning_decision = "exploiting"
        
        self.actions_taken = {"throttle": throttle, "brake": brake, "steering": steering}
        return [throttle, brake, steering]
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0, 0.0]
        
        # Q-learning update equation
        current_q = self.q_table[state]
        next_q = self.q_table[next_state]
        max_next_q = max(next_q)
        
        # Update each action component and track policy loss
        total_policy_loss = 0.0
        for i in range(3):
            old_q = current_q[i]
            current_q[i] += self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q[i]
            )
            policy_loss = abs(current_q[i] - old_q)
            total_policy_loss += policy_loss
        
        # Store learning metrics
        self.policy_losses.append(total_policy_loss / 3.0)
        self.value_estimates.append(max(current_q))
        
        # Decay exploration rate (transition from random to learned behavior)
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        self.total_steps += 1
        self.current_reward = reward
        
        # Store learning progress
        if len(self.episode_rewards) > 0:
            avg_recent = np.mean(list(self.episode_rewards)[-10:])
            self.learning_curve.append(avg_recent)
    
    def complete_episode(self):
        """Complete an episode and update metrics"""
        self.episode += 1
        self.episode_rewards.append(self.current_reward)
        self.total_races += 1
        
        # Check if this was a winning performance
        if self.current_reward > np.mean(list(self.episode_rewards)[-5:]) if len(self.episode_rewards) > 5 else 0:
            self.races_won += 1
        
        logger.info(f"Agent {self.name} completed episode {self.episode}, reward: {self.current_reward:.2f}")
    
    def get_learning_metrics(self):
        """Get comprehensive learning metrics"""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_q_value = np.mean(list(self.q_values_history)[-20:]) if self.q_values_history else 0.0
        avg_policy_loss = np.mean(list(self.policy_losses)[-20:]) if self.policy_losses else 0.0
        avg_value_estimate = np.mean(list(self.value_estimates)[-20:]) if self.value_estimates else 0.0
        win_rate = (self.races_won / self.total_races * 100) if self.total_races > 0 else 0.0
        
        # Calculate learning progress indicators
        learning_progress = min(100, (self.episode / 50.0) * 100) if self.episode > 0 else 0.0
        convergence_rate = (1.0 - self.exploration_rate) * 100
        
        return {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "exploration_rate": round(self.exploration_rate, 4),
            "learning_rate": self.learning_rate,
            "q_states": len(self.q_table),
            "avg_reward": round(avg_reward, 2),
            "current_reward": round(self.current_reward, 2),
            "avg_q_value": round(avg_q_value, 4),
            "avg_policy_loss": round(avg_policy_loss, 4),
            "avg_value_estimate": round(avg_value_estimate, 4),
            "win_rate": round(win_rate, 1),
            "learning_progress": round(learning_progress, 1),
            "convergence_rate": round(convergence_rate, 1),
            "learning_decision": getattr(self, 'learning_decision', 'exploring'),
            "races_won": self.races_won,
            "total_races": self.total_races,
            "training_time": self.training_time,
            
            # Historical data for charts
            "reward_history": list(self.episode_rewards),
            "q_value_history": list(self.q_values_history)[-50:],
            "policy_loss_history": list(self.policy_losses)[-50:],
            "value_estimate_history": list(self.value_estimates)[-50:]
        }

class Track3D:
    """3D Racing track"""
    
    def __init__(self):
        self.waypoints = []
        self.generate_track()
    
    def generate_track(self):
        """Generate simple racing track"""
        num_points = 100
        for i in range(num_points):
            progress = i / num_points
            angle = progress * 2 * math.pi
            
            # Simple oval track with elevation
            radius = 200 + 50 * math.sin(progress * 4 * math.pi)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 20 * math.sin(progress * 2 * math.pi)
            
            heading = angle + math.pi/2
            
            self.waypoints.append({
                'x': x, 'y': y, 'z': z,
                'heading': heading,
                'progress': progress
            })
    
    def get_waypoint(self, position: float):
        """Get waypoint at normalized position"""
        index = int(position * len(self.waypoints)) % len(self.waypoints)
        return self.waypoints[index]

class RacingCar3D:
    """3D Racing car connected to RL agent"""
    
    def __init__(self, agent: QLearningAgent, track: Track3D):
        self.agent = agent
        self.track = track
        
        # Physical properties
        self.speed = 0.0
        self.max_speed = 200 + random.uniform(-20, 20)
        self.acceleration = 0.0
        
        # Racing state
        self.lap_count = 0
        self.lap_start_time = time.time()
        self.last_state = None
        self.last_action = None
        
    def update_physics_with_rl(self, dt: float = 0.05):
        """Update car physics based on RL agent decisions"""
        old_position = self.agent.track_position
        
        # Get current state for RL
        competitors_nearby = random.randint(0, 2)  # Simple simulation
        current_state = self.agent.get_state(
            self.agent.track_position, 
            self.speed, 
            competitors_nearby
        )
        
        # Get action from RL agent
        action = self.agent.get_action(current_state)
        throttle, brake, steering = action
        
        # Apply physics based on RL decisions
        engine_force = throttle * 10.0
        brake_force = brake * 15.0
        air_resistance = 0.001 * self.speed * self.speed
        
        net_force = engine_force - brake_force - air_resistance
        self.acceleration = net_force
        self.speed += self.acceleration * dt
        self.speed = max(0, min(self.max_speed, self.speed))
        
        # Update track position
        track_length = 2000  # meters
        speed_ms = self.speed / 3.6
        position_delta = (speed_ms * dt) / track_length
        self.agent.track_position = (self.agent.track_position + position_delta) % 1.0
        
        # Update lateral position based on steering
        self.agent.lateral_offset += steering * self.speed * dt * 0.01
        self.agent.lateral_offset = max(-10, min(10, self.agent.lateral_offset))
        
        # Calculate reward based on racing performance
        reward = self.calculate_racing_reward(old_position)
        
        # Update Q-learning if we have previous state
        if self.last_state is not None and self.last_action is not None:
            self.agent.update_q_table(self.last_state, self.last_action, reward, current_state)
        
        self.last_state = current_state
        self.last_action = action
        
        # Check lap completion
        if old_position > 0.9 and self.agent.track_position < 0.1:
            self.complete_lap()
        
        return self.get_3d_position()
    
    def calculate_racing_reward(self, old_position: float):
        """Calculate reward for RL learning"""
        # Progress reward
        progress_delta = self.agent.track_position - old_position
        if progress_delta < -0.5:  # Lap completion
            progress_delta += 1.0
        progress_reward = progress_delta * 1000
        
        # Speed efficiency reward
        speed_efficiency = min(1.0, self.speed / 150.0)  # Target speed ~150 km/h
        speed_reward = speed_efficiency * 50
        
        # Racing line reward (stay near center)
        line_penalty = abs(self.agent.lateral_offset) * 5
        
        # Smoothness reward (avoid erratic steering)
        steering_penalty = abs(self.agent.actions_taken["steering"]) * 10
        
        total_reward = progress_reward + speed_reward - line_penalty - steering_penalty
        return max(-50, min(200, total_reward))
    
    def complete_lap(self):
        """Handle lap completion"""
        current_time = time.time()
        lap_time = current_time - self.lap_start_time
        
        if lap_time < self.agent.best_lap_time:
            self.agent.best_lap_time = lap_time
        
        self.agent.current_lap_time = lap_time
        self.lap_count += 1
        self.lap_start_time = current_time
        
        # Complete episode every few laps
        if self.lap_count % 3 == 0:
            self.agent.complete_episode()
    
    def get_3d_position(self):
        """Get 3D world position for visualization"""
        waypoint = self.track.get_waypoint(self.agent.track_position)
        
        # Calculate position with lateral offset
        heading = waypoint['heading']
        offset_x = self.agent.lateral_offset * math.cos(heading + math.pi/2)
        offset_y = self.agent.lateral_offset * math.sin(heading + math.pi/2)
        
        return {
            'x': waypoint['x'] + offset_x,
            'y': waypoint['y'] + offset_y,
            'z': waypoint['z'] + 3.0,
            'heading': heading + self.agent.actions_taken["steering"] * 0.3,
            'track_position': self.agent.track_position
        }

class SynchronizedRLSystem:
    """Synchronized RL Learning + 3D Racing System"""
    
    def __init__(self):
        self.track = Track3D()
        self.agents = {}
        self.cars = {}
        self.simulation_active = False
        self.start_time = None
        self.training_start_time = None
        self.episode_data = deque(maxlen=100)
        
    def initialize_system(self):
        """Initialize RL agents and 3D cars"""
        agent_configs = [
            {"id": "rl_smart", "name": "🧠 Smart Learner", "color": "#3498db", "lr": 0.12},
            {"id": "rl_aggressive", "name": "🔥 Aggressive Learner", "color": "#e74c3c", "lr": 0.15},
            {"id": "rl_cautious", "name": "🛡️ Cautious Learner", "color": "#2ecc71", "lr": 0.08}
        ]
        
        for config in agent_configs:
            # Create RL agent
            agent = QLearningAgent(
                config["id"], 
                config["name"], 
                config["color"],
                config["lr"]
            )
            self.agents[config["id"]] = agent
            
            # Create 3D car linked to agent
            car = RacingCar3D(agent, self.track)
            self.cars[config["id"]] = car
        
        logger.info(f"Initialized {len(self.agents)} RL agents with synchronized 3D cars")
    
    def start_simulation(self):
        """Start synchronized RL + 3D simulation"""
        if self.simulation_active:
            return False
        
        self.simulation_active = True
        self.start_time = time.time()
        self.training_start_time = time.time()
        
        if not self.agents:
            self.initialize_system()
        
        # Start simulation loop
        asyncio.create_task(self.simulation_loop())
        logger.info("Synchronized RL + 3D simulation started")
        return True
    
    def stop_simulation(self):
        """Stop simulation"""
        self.simulation_active = False
        logger.info("Simulation stopped")
    
    async def simulation_loop(self):
        """Main simulation loop"""
        while self.simulation_active:
            dt = 0.05  # 20 FPS
            
            # Update all cars with RL decisions
            for car in self.cars.values():
                car.update_physics_with_rl(dt)
            
            # Store data every second
            if int(time.time() * 1) % 1 == 0:
                self.store_episode_data()
            
            await asyncio.sleep(dt)
    
    def store_episode_data(self):
        """Store synchronized learning and racing data"""
        current_time = time.time() - self.start_time if self.start_time else 0
        
        episode_data = {
            "timestamp": current_time,
            "agents": {}
        }
        
        for agent_id, agent in self.agents.items():
            # Update training time
            if self.training_start_time:
                agent.training_time = current_time
            
            metrics = agent.get_learning_metrics()
            position = self.cars[agent_id].get_3d_position()
            
            episode_data["agents"][agent_id] = {
                **metrics,
                "position": position,
                "speed": self.cars[agent_id].speed,
                "actions": agent.actions_taken
            }
        
        self.episode_data.append(episode_data)
    
    def get_synchronized_data(self):
        """Get complete synchronized RL + 3D data"""
        simulation_time = time.time() - self.start_time if self.start_time else 0
        
        synchronized_data = []
        for agent_id, agent in self.agents.items():
            car = self.cars[agent_id]
            position = car.get_3d_position()
            metrics = agent.get_learning_metrics()
            
            synchronized_data.append({
                'id': agent_id,
                'name': agent.name,
                'color': agent.color,
                
                # 3D Racing Data
                'position': position,
                'speed': car.speed,
                'lap_count': car.lap_count,
                'track_position': agent.track_position,
                
                # RL Learning Data
                'rl_metrics': metrics,
                'current_actions': agent.actions_taken,
                'learning_decision': getattr(agent, 'learning_decision', 'exploring'),
                
                # Combined Performance
                'performance_score': metrics['avg_reward'],
                'learning_progress': min(100, (agent.episode / 50) * 100)
            })
        
        # Sort by current performance
        synchronized_data.sort(key=lambda x: x['performance_score'], reverse=True)
        for i, data in enumerate(synchronized_data):
            data['race_position'] = i + 1
        
        return {
            'simulation_active': self.simulation_active,
            'simulation_time': simulation_time,
            'cars': synchronized_data,
            'track': self.get_track_data(),
            'learning_summary': self.get_learning_summary(),
            'episode_history': list(self.episode_data)[-20:],  # Last 20 data points
            'timestamp': datetime.now().isoformat()
        }
    
    def get_track_data(self):
        """Get track data for 3D rendering"""
        return [
            {
                'x': wp['x'], 'y': wp['y'], 'z': wp['z'],
                'heading': wp['heading'], 'progress': wp['progress']
            }
            for wp in self.track.waypoints[::2]  # Every other waypoint
        ]
    
    def get_learning_summary(self):
        """Get learning progress summary"""
        if not self.agents:
            return {}
        
        total_episodes = sum(agent.episode for agent in self.agents.values())
        avg_exploration = np.mean([agent.exploration_rate for agent in self.agents.values()])
        total_q_states = sum(len(agent.q_table) for agent in self.agents.values())
        
        return {
            'total_episodes': total_episodes,
            'avg_exploration_rate': round(avg_exploration, 4),
            'total_q_states': total_q_states,
            'learning_stage': 'exploring' if avg_exploration > 0.5 else 'exploiting',
            'system_performance': 'improving' if total_episodes > 10 else 'initializing'
        }

# Global system
rl_system = SynchronizedRLSystem()

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# FastAPI app
app = FastAPI(title="Synchronized RL + 3D Racing System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time synchronized data"""
    await manager.connect(websocket)
    try:
        while True:
            data = rl_system.get_synchronized_data()
            await websocket.send_json(data)
            await asyncio.sleep(0.1)  # 10 FPS for smooth updates
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/simulation/start")
async def start_simulation():
    """Start synchronized RL + 3D simulation"""
    success = rl_system.start_simulation()
    return {"status": "started" if success else "already_running"}

@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop simulation"""
    rl_system.stop_simulation()
    return {"status": "stopped"}

@app.get("/api/status")
async def get_status():
    """Get synchronized system status"""
    return rl_system.get_synchronized_data()

@app.get("/")
async def root():
    """Synchronized RL + 3D visualization interface"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Synchronized RL Learning + 3D Racing</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            margin: 0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: #000; 
            color: white; 
            overflow: hidden;
        }
        
        #container {
            display: flex;
            width: 100vw;
            height: 100vh;
        }
        
        #left-panel {
            width: 35%;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            overflow-y: auto;
            border-right: 2px solid #3498db;
        }
        
        #racing-canvas {
            width: 65%;
            height: 100vh;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .agent-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        
        .metric-value {
            font-weight: bold;
            color: #3498db;
        }
        
        .learning-indicator {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .exploring { background: #e74c3c; }
        .exploiting { background: #2ecc71; }
        
        .chart-container {
            height: 180px;
            margin: 10px 0;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 8px;
        }
        
        .chart-container h4 {
            margin: 0 0 8px 0;
            font-size: 0.9em;
            color: #3498db;
        }
        
        .mini-chart {
            height: 120px;
            margin: 8px 0;
            background: rgba(255,255,255,0.03);
            border-radius: 6px;
            padding: 6px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin: 10px 0;
        }
        
        .metric-card {
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            padding: 8px;
            text-align: center;
        }
        
        .metric-card .value {
            font-size: 1.2em;
            font-weight: bold;
            color: #3498db;
        }
        
        .metric-card .label {
            font-size: 0.8em;
            opacity: 0.8;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active { background: #2ecc71; animation: pulse 2s infinite; }
        .status-inactive { background: #e74c3c; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .controls {
            position: absolute;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            cursor: pointer;
            margin: 0 5px;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        
        .btn:disabled {
            background: #7f8c8d;
            cursor: not-allowed;
            transform: none;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="left-panel">
            <div class="header">
                <h3>🧠 RL Learning + 🏎️ 3D Racing</h3>
                <div>
                    <span class="status-indicator" id="statusIndicator"></span>
                    <span id="systemStatus">Connecting...</span>
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="value" id="totalEpisodes">0</div>
                    <div class="label">Episodes</div>
                </div>
                <div class="metric-card">
                    <div class="value" id="avgExploration">100%</div>
                    <div class="label">Exploration</div>
                </div>
                <div class="metric-card">
                    <div class="value" id="totalQStates">0</div>
                    <div class="label">Q-States</div>
                </div>
                <div class="metric-card">
                    <div class="value" id="trainingTime">0:00</div>
                    <div class="label">Training Time</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h4>📈 Learning Curves (Rewards)</h4>
                <canvas id="rewardChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h4>🎯 Exploration vs Exploitation</h4>
                <canvas id="explorationChart"></canvas>
            </div>
            
            <div class="mini-chart">
                <h4>🧠 Q-Value Evolution</h4>
                <canvas id="qValueChart"></canvas>
            </div>
            
            <div class="mini-chart">
                <h4>📉 Policy Loss</h4>
                <canvas id="policyLossChart"></canvas>
            </div>
            
            <div id="agentCards"></div>
            
            <div class="mini-chart">
                <h4>💡 Value Estimates</h4>
                <canvas id="valueEstimateChart"></canvas>
            </div>
        </div>
        
        <canvas id="racing-canvas"></canvas>
    </div>
    
    <div class="controls">
        <button class="btn" id="startBtn" onclick="startSimulation()">🚀 Start RL + 3D</button>
        <button class="btn" id="stopBtn" onclick="stopSimulation()" disabled>⏹️ Stop</button>
        <button class="btn" onclick="resetCamera()">📷 Reset Camera</button>
    </div>

    <script>
        let scene, camera, renderer, controls;
        let cars = {};
        let trackMesh;
        let ws = null;
        let rewardChart, explorationChart, qValueChart, policyLossChart, valueEstimateChart;
        
        // Initialize 3D scene
        function init3DScene() {
            const canvas = document.getElementById('racing-canvas');
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x222222);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 1, 5000);
            camera.position.set(0, 300, 500);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            renderer.shadowMap.enabled = true;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(200, 300, 100);
            directionalLight.castShadow = true;
            scene.add(directionalLight);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            animate();
        }
        
        function initCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: 'white' } },
                    x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: 'white' } }
                },
                plugins: { legend: { labels: { color: 'white', font: { size: 10 } } } }
            };
            
            // Reward Chart
            const rewardCtx = document.getElementById('rewardChart').getContext('2d');
            rewardChart = new Chart(rewardCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: chartOptions
            });
            
            // Q-Value Chart
            const qValueCtx = document.getElementById('qValueChart').getContext('2d');
            qValueChart = new Chart(qValueCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: chartOptions
            });
            
            // Policy Loss Chart
            const policyLossCtx = document.getElementById('policyLossChart').getContext('2d');
            policyLossChart = new Chart(policyLossCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: chartOptions
            });
            
            // Value Estimate Chart
            const valueEstimateCtx = document.getElementById('valueEstimateChart').getContext('2d');
            valueEstimateChart = new Chart(valueEstimateCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: chartOptions
            });
            
            // Exploration Chart
            const explorationCtx = document.getElementById('explorationChart').getContext('2d');
            explorationChart = new Chart(explorationCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Exploration', 'Exploitation'],
                    datasets: [{
                        data: [50, 50],
                        backgroundColor: ['#e74c3c', '#2ecc71']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: 'white', font: { size: 10 } } } }
                }
            });
        }
        
        function createTrack(trackData) {
            if (trackMesh) {
                scene.remove(trackMesh);
            }
            
            const geometry = new THREE.BufferGeometry();
            const positions = [];
            
            for (let i = 0; i < trackData.length - 1; i++) {
                const current = trackData[i];
                const next = trackData[i + 1];
                
                // Create track segment
                positions.push(
                    current.x, current.z, current.y,
                    next.x, next.z, next.y
                );
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            const material = new THREE.LineBasicMaterial({ color: 0x3498db, linewidth: 3 });
            trackMesh = new THREE.Line(geometry, material);
            scene.add(trackMesh);
        }
        
        function createCar(carData) {
            const carGroup = new THREE.Group();
            
            // Car body - color based on RL agent
            const bodyGeometry = new THREE.BoxGeometry(8, 3, 16);
            const bodyMaterial = new THREE.MeshLambertMaterial({ color: carData.color });
            const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
            body.castShadow = true;
            carGroup.add(body);
            
            // Learning indicator (glows when exploiting, dims when exploring)
            const indicatorGeometry = new THREE.SphereGeometry(2, 8, 8);
            const indicatorMaterial = new THREE.MeshLambertMaterial({ 
                color: carData.learning_decision === 'exploiting' ? 0x00ff00 : 0xff0000,
                transparent: true,
                opacity: carData.learning_decision === 'exploiting' ? 0.8 : 0.3
            });
            const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
            indicator.position.set(0, 6, 0);
            carGroup.add(indicator);
            
            return carGroup;
        }
        
        function updateCars(carsData) {
            carsData.forEach(carData => {
                if (!cars[carData.id]) {
                    cars[carData.id] = createCar(carData);
                    scene.add(cars[carData.id]);
                }
                
                const carMesh = cars[carData.id];
                const pos = carData.position;
                
                // Update position
                carMesh.position.set(pos.x, pos.z + 3, pos.y);
                carMesh.rotation.y = -pos.heading;
                
                // Update learning indicator
                const indicator = carMesh.children[1];
                indicator.material.color.setHex(
                    carData.learning_decision === 'exploiting' ? 0x00ff00 : 0xff0000
                );
                indicator.material.opacity = carData.learning_decision === 'exploiting' ? 0.8 : 0.3;
            });
        }
        
        function updateLearningPanels(data) {
            // Update status
            document.getElementById('systemStatus').textContent = 
                data.simulation_active ? 'Learning & Racing Active' : 'Stopped';
            document.getElementById('statusIndicator').className = 
                'status-indicator ' + (data.simulation_active ? 'status-active' : 'status-inactive');
            
            // Update top metrics
            if (data.cars.length > 0) {
                const totalEpisodes = Math.max(...data.cars.map(c => c.rl_metrics.episode));
                const avgExploration = data.cars.reduce((sum, c) => sum + c.rl_metrics.exploration_rate, 0) / data.cars.length;
                const totalQStates = data.cars.reduce((sum, c) => sum + c.rl_metrics.q_states, 0);
                const trainingTime = Math.max(...data.cars.map(c => c.rl_metrics.training_time || 0));
                
                document.getElementById('totalEpisodes').textContent = totalEpisodes;
                document.getElementById('avgExploration').textContent = (avgExploration * 100).toFixed(1) + '%';
                document.getElementById('totalQStates').textContent = totalQStates;
                
                const minutes = Math.floor(trainingTime / 60);
                const seconds = Math.floor(trainingTime % 60);
                document.getElementById('trainingTime').textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }
            
            // Update agent cards
            const container = document.getElementById('agentCards');
            container.innerHTML = '';
            
            data.cars.forEach(car => {
                const card = document.createElement('div');
                card.className = 'agent-card';
                card.style.borderLeftColor = car.color;
                
                const metrics = car.rl_metrics;
                card.innerHTML = `
                    <h4 style="font-size: 0.9em;">${car.name} <span class="learning-indicator ${car.learning_decision}">${car.learning_decision}</span></h4>
                    <div class="metric-row">
                        <span>Episode:</span>
                        <span class="metric-value">${metrics.episode}</span>
                    </div>
                    <div class="metric-row">
                        <span>Convergence:</span>
                        <span class="metric-value">${metrics.convergence_rate}%</span>
                    </div>
                    <div class="metric-row">
                        <span>Q-States:</span>
                        <span class="metric-value">${metrics.q_states}</span>
                    </div>
                    <div class="metric-row">
                        <span>Avg Reward:</span>
                        <span class="metric-value">${metrics.avg_reward.toFixed(0)}</span>
                    </div>
                    <div class="metric-row">
                        <span>Policy Loss:</span>
                        <span class="metric-value">${metrics.avg_policy_loss.toFixed(3)}</span>
                    </div>
                    <div class="metric-row">
                        <span>Speed:</span>
                        <span class="metric-value">${car.speed.toFixed(0)} km/h</span>
                    </div>
                `;
                
                container.appendChild(card);
            });
            
            // Update charts
            updateCharts(data);
        }
        
        function updateCharts(data) {
            if (data.cars.length > 0) {
                // Update exploration chart
                const avgExploration = data.cars.reduce((sum, car) => 
                    sum + car.rl_metrics.exploration_rate, 0) / data.cars.length;
                
                explorationChart.data.datasets[0].data = [
                    avgExploration * 100,
                    (1 - avgExploration) * 100
                ];
                explorationChart.update('none');
                
                // Update all line charts with historical data
                data.cars.forEach((car, index) => {
                    const metrics = car.rl_metrics;
                    const color = car.color;
                    
                    // Reward Chart
                    if (metrics.reward_history && metrics.reward_history.length > 0) {
                        updateLineChart(rewardChart, index, car.name, metrics.reward_history, color);
                    }
                    
                    // Q-Value Chart
                    if (metrics.q_value_history && metrics.q_value_history.length > 0) {
                        updateLineChart(qValueChart, index, car.name, metrics.q_value_history, color);
                    }
                    
                    // Policy Loss Chart
                    if (metrics.policy_loss_history && metrics.policy_loss_history.length > 0) {
                        updateLineChart(policyLossChart, index, car.name, metrics.policy_loss_history, color);
                    }
                    
                    // Value Estimate Chart
                    if (metrics.value_estimate_history && metrics.value_estimate_history.length > 0) {
                        updateLineChart(valueEstimateChart, index, car.name, metrics.value_estimate_history, color);
                    }
                });
            }
        }
        
        function updateLineChart(chart, index, label, data, color) {
            // Ensure dataset exists
            if (!chart.data.datasets[index]) {
                chart.data.datasets[index] = {
                    label: label,
                    data: [],
                    borderColor: color,
                    backgroundColor: color + '20',
                    tension: 0.1,
                    fill: false,
                    borderWidth: 2,
                    pointRadius: 0
                };
            }
            
            // Update data and labels
            chart.data.labels = data.map((_, i) => i);
            chart.data.datasets[index].data = data;
            chart.update('none');
        }
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.track && !trackMesh) {
                    createTrack(data.track);
                }
                
                if (data.cars) {
                    updateCars(data.cars);
                    updateLearningPanels(data);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected. Reconnecting...');
                setTimeout(connectWebSocket, 1000);
            };
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            if (controls) {
                controls.update();
            }
            
            renderer.render(scene, camera);
        }
        
        function resetCamera() {
            camera.position.set(0, 300, 500);
            camera.lookAt(0, 0, 0);
            controls.reset();
        }
        
        async function startSimulation() {
            try {
                const response = await fetch('/api/simulation/start', { method: 'POST' });
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            } catch (error) {
                console.error('Error starting simulation:', error);
            }
        }
        
        async function stopSimulation() {
            try {
                const response = await fetch('/api/simulation/stop', { method: 'POST' });
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            } catch (error) {
                console.error('Error stopping simulation:', error);
            }
        }
        
        // Initialize everything
        init3DScene();
        initCharts();
        connectWebSocket();
        
        // Auto-start for demo
        setTimeout(() => {
            startSimulation();
        }, 2000);
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    logger.info("Starting Synchronized RL + 3D Racing System...")
    uvicorn.run(app, host="0.0.0.0", port=7001, log_level="info")