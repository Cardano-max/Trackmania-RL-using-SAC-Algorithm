#!/usr/bin/env python3
"""
Complete TrackMania RL + 3D Visualization System
Full implementation with RL learning and 3D racing visualization
"""

import asyncio
import json
import time
import math
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAgent:
    """Complete RL Agent with Q-Learning"""
    
    def __init__(self, agent_id: str, name: str, color: str):
        self.agent_id = agent_id
        self.name = name
        self.color = color
        
        # RL Parameters
        self.q_table = {}
        self.exploration_rate = 1.0
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.min_exploration = 0.05
        self.exploration_decay = 0.995
        
        # Performance tracking
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.q_values_history = deque(maxlen=1000)
        self.policy_losses = deque(maxlen=100)
        
        # Racing metrics
        self.current_reward = 0.0
        self.best_lap_time = float('inf')
        self.current_lap_time = 0.0
        self.sector_times = [0.0, 0.0, 0.0, 0.0]
        
    def get_state(self, track_position: float, speed: float, distance_to_line: float, 
                  next_turn_distance: float, next_turn_sharpness: float):
        """Discretize state for Q-learning"""
        pos_bucket = int(track_position * 20) % 20
        speed_bucket = min(4, int(speed / 50))
        line_bucket = min(3, int(abs(distance_to_line) / 10))
        turn_dist_bucket = min(3, int(next_turn_distance / 50))
        turn_sharp_bucket = min(2, int(next_turn_sharpness))
        
        return (pos_bucket, speed_bucket, line_bucket, turn_dist_bucket, turn_sharp_bucket)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]  # [throttle, brake, steering]
        
        if random.random() < self.exploration_rate:
            # Exploration: random action
            throttle = random.uniform(0.3, 1.0)
            brake = random.uniform(0.0, 0.5)
            steering = random.uniform(-1.0, 1.0)
        else:
            # Exploitation: use Q-values
            q_values = self.q_table[state]
            self.q_values_history.append(max(q_values))
            
            # Convert Q-values to actions
            throttle = max(0.0, min(1.0, 0.7 + q_values[0] * 0.3))
            brake = max(0.0, min(1.0, q_values[1]))
            steering = max(-1.0, min(1.0, q_values[2]))
        
        return [throttle, brake, steering]
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0, 0.0]
        
        # Q-learning update
        current_q = self.q_table[state]
        next_q = self.q_table[next_state]
        max_next_q = max(next_q)
        
        # Update each action component
        for i in range(3):
            old_q = current_q[i]
            current_q[i] += self.learning_rate * (reward + self.discount_factor * max_next_q - current_q[i])
            
            # Track policy loss
            policy_loss = abs(current_q[i] - old_q)
            self.policy_losses.append(policy_loss)
        
        # Decay exploration
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        
        self.total_steps += 1
    
    def get_metrics(self):
        """Get comprehensive RL metrics"""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_q_value = np.mean(list(self.q_values_history)[-50:]) if self.q_values_history else 0.0
        avg_policy_loss = np.mean(list(self.policy_losses)[-20:]) if self.policy_losses else 0.0
        
        return {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'q_states': len(self.q_table),
            'avg_reward': avg_reward,
            'current_reward': self.current_reward,
            'avg_q_value': avg_q_value,
            'policy_loss': avg_policy_loss,
            'best_lap_time': self.best_lap_time if self.best_lap_time != float('inf') else 0.0
        }

class Track3D:
    """Advanced 3D track with realistic racing features"""
    
    def __init__(self):
        self.waypoints = []
        self.sector_markers = []
        self.generate_professional_track()
    
    def generate_professional_track(self):
        """Generate complex racing track with multiple sections"""
        num_points = 300
        
        for i in range(num_points):
            progress = i / num_points
            
            # Complex track sections
            if progress < 0.2:  # High-speed straight
                x = progress * 1000 - 500
                y = 0
                z = 0
                banking = 0
                speed_limit = 250
                turn_sharpness = 0
            elif progress < 0.35:  # Hairpin turn
                local_progress = (progress - 0.2) / 0.15
                angle = local_progress * math.pi
                radius = 80
                x = 500 + radius * math.cos(angle - math.pi/2)
                y = radius * math.sin(angle - math.pi/2)
                z = 20 * math.sin(local_progress * math.pi)
                banking = -15
                speed_limit = 80
                turn_sharpness = 2
            elif progress < 0.5:  # Climbing section
                local_progress = (progress - 0.35) / 0.15
                x = 500 - local_progress * 200
                y = 80 + local_progress * 100
                z = 20 + local_progress * 60
                banking = 5
                speed_limit = 160
                turn_sharpness = 1
            elif progress < 0.65:  # Chicane complex
                local_progress = (progress - 0.5) / 0.15
                x = 300 - local_progress * 600
                y = 180 + 120 * math.sin(local_progress * 4 * math.pi)
                z = 80 - local_progress * 40
                banking = 8 * math.sin(local_progress * 4 * math.pi)
                speed_limit = 140
                turn_sharpness = 1.5
            elif progress < 0.8:  # Fast sweeper
                local_progress = (progress - 0.65) / 0.15
                angle = local_progress * math.pi * 1.5
                radius = 200
                x = -300 + radius * math.cos(angle)
                y = radius * math.sin(angle) + 100
                z = 40 - local_progress * 30
                banking = 12
                speed_limit = 200
                turn_sharpness = 0.5
            else:  # Return straight
                local_progress = (progress - 0.8) / 0.2
                x = -300 - local_progress * 200
                y = 0
                z = 10 - local_progress * 10
                banking = 0
                speed_limit = 220
                turn_sharpness = 0
            
            # Calculate heading
            next_i = (i + 1) % num_points
            next_progress = next_i / num_points
            next_x, next_y = self.get_position(next_progress)
            
            heading = math.atan2(next_y - y, next_x - x)
            
            # Calculate racing line offset
            racing_line_offset = 0
            if turn_sharpness > 1:
                racing_line_offset = -banking * 0.5
            
            self.waypoints.append({
                'x': x, 'y': y, 'z': z,
                'heading': heading,
                'banking': banking,
                'speed_limit': speed_limit,
                'turn_sharpness': turn_sharpness,
                'racing_line_offset': racing_line_offset,
                'sector': int(progress * 4),
                'checkpoint': i % 25 == 0,
                'start_finish': i == 0
            })
        
        # Add sector markers
        for sector in range(4):
            sector_point = int(sector * num_points / 4)
            self.sector_markers.append(sector_point)
    
    def get_position(self, progress):
        """Get basic x,y position for heading calculation"""
        if progress < 0.2:
            return progress * 1000 - 500, 0
        elif progress < 0.35:
            local_progress = (progress - 0.2) / 0.15
            angle = local_progress * math.pi
            return 500 + 80 * math.cos(angle - math.pi/2), 80 * math.sin(angle - math.pi/2)
        # Simplified for other sections
        return 0, 0
    
    def get_waypoint(self, position: float):
        """Get waypoint at normalized position (0-1)"""
        index = int(position * len(self.waypoints)) % len(self.waypoints)
        return self.waypoints[index]
    
    def get_next_turn_info(self, position: float):
        """Get information about upcoming turn"""
        current_index = int(position * len(self.waypoints))
        
        # Look ahead for next significant turn
        for i in range(20):  # Look ahead 20 waypoints
            check_index = (current_index + i) % len(self.waypoints)
            waypoint = self.waypoints[check_index]
            
            if waypoint['turn_sharpness'] > 1.0:
                distance = i * 20  # Approximate distance in meters
                return distance, waypoint['turn_sharpness']
        
        return 200, 0  # Default: no significant turn ahead

class RacingCar3D:
    """Advanced 3D racing car with RL integration"""
    
    def __init__(self, agent: RLAgent, track: Track3D):
        self.agent = agent
        self.track = track
        
        # Physical properties
        self.track_position = random.uniform(0, 0.05)  # Start near start line
        self.speed = 0.0  # km/h
        self.lateral_offset = random.uniform(-3, 3)
        self.heading_offset = 0.0
        
        # Car dynamics
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        self.gear = 1
        self.rpm = 1000
        
        # Racing state
        self.lap_count = 0
        self.lap_start_time = None
        self.sector_times = [0.0, 0.0, 0.0, 0.0]
        self.current_sector = 0
        self.episode_start_time = time.time()
        
        # Physics parameters
        self.max_speed = 280
        self.acceleration_rate = 12.0
        self.braking_rate = 18.0
        self.steering_sensitivity = 0.8
        
        # RL training state
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
    
    def get_current_state(self):
        """Get current state for RL agent"""
        waypoint = self.track.get_waypoint(self.track_position)
        next_turn_distance, next_turn_sharpness = self.track.get_next_turn_info(self.track_position)
        
        return self.agent.get_state(
            self.track_position,
            self.speed,
            abs(self.lateral_offset),
            next_turn_distance,
            next_turn_sharpness
        )
    
    def calculate_reward(self, old_position: float):
        """Calculate RL reward based on racing performance"""
        waypoint = self.track.get_waypoint(self.track_position)
        
        # Progress reward (most important)
        progress_delta = self.track_position - old_position
        if progress_delta < -0.5:  # Handle lap completion
            progress_delta += 1.0
        progress_reward = progress_delta * 1000
        
        # Speed efficiency reward
        target_speed = waypoint['speed_limit']
        speed_efficiency = 1.0 - abs(self.speed - target_speed) / target_speed
        speed_reward = speed_efficiency * 50
        
        # Racing line reward
        ideal_lateral = waypoint['racing_line_offset']
        line_error = abs(self.lateral_offset - ideal_lateral)
        line_reward = max(0, 30 - line_error * 2)
        
        # Smoothness reward
        steering_penalty = abs(self.steering) * 10
        
        # Sector time bonus
        sector_bonus = 0
        if self.current_sector != int(self.track_position * 4):
            sector_bonus = 20
        
        total_reward = (
            progress_reward * 0.4 +
            speed_reward * 0.25 +
            line_reward * 0.2 +
            sector_bonus * 0.1 -
            steering_penalty * 0.05
        )
        
        return max(-50, min(100, total_reward))
    
    def update_physics(self, dt: float = 0.05):
        """Update car physics and RL learning"""
        old_position = self.track_position
        
        # Get current state for RL
        current_state = self.get_current_state()
        
        # Get action from RL agent
        action = self.agent.get_action(current_state)
        self.throttle, self.brake, self.steering = action
        
        # Apply physics
        waypoint = self.track.get_waypoint(self.track_position)
        
        # Engine and braking forces
        engine_force = self.throttle * self.acceleration_rate
        brake_force = self.brake * self.braking_rate
        air_resistance = 0.001 * self.speed * self.speed
        
        # Calculate acceleration
        net_force = engine_force - brake_force - air_resistance
        acceleration = net_force
        
        # Update speed
        self.speed += acceleration * dt
        self.speed = max(0, min(self.max_speed, self.speed))
        
        # Update track position
        track_length = 4000  # meters
        speed_ms = self.speed / 3.6
        position_delta = (speed_ms * dt) / track_length
        self.track_position = (self.track_position + position_delta) % 1.0
        
        # Update lateral position (steering)
        self.lateral_offset += self.steering * self.speed * dt * 0.02
        self.lateral_offset = max(-15, min(15, self.lateral_offset))
        
        # Track banking effect
        banking_effect = waypoint['banking'] * 0.1
        self.lateral_offset += banking_effect * dt
        
        # Update lap tracking
        new_sector = int(self.track_position * 4)
        if new_sector != self.current_sector:
            self.current_sector = new_sector
        
        # Check lap completion
        if old_position > 0.95 and self.track_position < 0.05:
            self.complete_lap()
        
        # Calculate reward and update RL
        reward = self.calculate_reward(old_position)
        self.agent.current_reward = reward
        
        if self.last_state is not None and self.last_action is not None:
            self.agent.update_q_table(self.last_state, self.last_action, reward, current_state)
        
        self.last_state = current_state
        self.last_action = action
        self.last_reward = reward
        
        # Update car systems
        self.rpm = max(800, min(8000, 1000 + self.speed * 80))
        self.gear = min(6, max(1, int(self.speed / 45) + 1))
        
        return self.get_world_position()
    
    def complete_lap(self):
        """Handle lap completion"""
        current_time = time.time()
        if self.lap_start_time:
            lap_time = current_time - self.lap_start_time
            self.agent.current_lap_time = lap_time
            
            if lap_time < self.agent.best_lap_time:
                self.agent.best_lap_time = lap_time
        
        self.lap_count += 1
        self.lap_start_time = current_time
        
        # Episode completion (every 3 laps)
        if self.lap_count % 3 == 0:
            self.agent.episode += 1
            self.agent.episode_rewards.append(self.agent.current_reward)
            logger.info(f"Agent {self.agent.name} completed episode {self.agent.episode}")
    
    def get_world_position(self):
        """Get 3D world position"""
        waypoint = self.track.get_waypoint(self.track_position)
        
        # Calculate lateral offset position
        heading = waypoint['heading']
        offset_x = self.lateral_offset * math.cos(heading + math.pi/2)
        offset_y = self.lateral_offset * math.sin(heading + math.pi/2)
        
        return {
            'x': waypoint['x'] + offset_x,
            'y': waypoint['y'] + offset_y,
            'z': waypoint['z'] + 3.0,  # Car height
            'heading': heading + self.steering * 0.2,
            'banking': waypoint['banking'],
            'track_position': self.track_position
        }

class CompleteRLSystem:
    """Complete RL + 3D Racing System"""
    
    def __init__(self):
        self.track = Track3D()
        self.cars = {}
        self.agents = {}
        self.simulation_active = False
        self.training_active = False
        self.start_time = None
        self.episode_data = []
        
    def initialize_agents(self):
        """Initialize RL agents and cars"""
        agent_configs = [
            {"id": "rl_aggressive", "name": "🔴 RL Aggressive", "color": "#e74c3c"},
            {"id": "rl_balanced", "name": "🔵 RL Balanced", "color": "#3498db"},
            {"id": "rl_cautious", "name": "🟢 RL Cautious", "color": "#2ecc71"},
        ]
        
        for config in agent_configs:
            # Create RL agent
            agent = RLAgent(config["id"], config["name"], config["color"])
            
            # Customize learning parameters
            if "aggressive" in config["id"]:
                agent.learning_rate = 0.15
                agent.exploration_decay = 0.99
            elif "cautious" in config["id"]:
                agent.learning_rate = 0.08
                agent.exploration_decay = 0.995
            
            self.agents[config["id"]] = agent
            
            # Create 3D racing car
            car = RacingCar3D(agent, self.track)
            self.cars[config["id"]] = car
        
        logger.info(f"Initialized {len(self.agents)} RL agents with 3D cars")
    
    def start_simulation(self):
        """Start complete RL + 3D simulation"""
        if self.simulation_active:
            return False
        
        self.simulation_active = True
        self.training_active = True
        self.start_time = time.time()
        
        if not self.agents:
            self.initialize_agents()
        
        # Start simulation loop
        asyncio.create_task(self.simulation_loop())
        logger.info("Complete RL + 3D simulation started")
        return True
    
    def stop_simulation(self):
        """Stop simulation"""
        self.simulation_active = False
        self.training_active = False
        logger.info("Simulation stopped")
    
    async def simulation_loop(self):
        """Main simulation loop with RL training"""
        while self.simulation_active:
            dt = 0.05  # 20 FPS
            
            # Update all cars (includes RL learning)
            for car in self.cars.values():
                car.update_physics(dt)
            
            # Store episode data every 20 frames (1 second)
            if int(time.time() * 20) % 20 == 0:
                self.store_episode_data()
            
            await asyncio.sleep(dt)
    
    def store_episode_data(self):
        """Store episode data for analysis"""
        episode_data = {
            "timestamp": time.time() - self.start_time if self.start_time else 0,
            "agents": {}
        }
        
        for agent_id, agent in self.agents.items():
            metrics = agent.get_metrics()
            episode_data["agents"][agent_id] = metrics
        
        self.episode_data.append(episode_data)
        
        # Keep only last 300 data points (15 seconds at 20Hz)
        if len(self.episode_data) > 300:
            self.episode_data = self.episode_data[-300:]
    
    def get_complete_data(self):
        """Get complete system data for visualization"""
        simulation_time = time.time() - self.start_time if self.start_time else 0
        
        # Car visualization data
        cars_data = []
        for car_id, car in self.cars.items():
            agent = self.agents[car_id]
            position = car.get_world_position()
            metrics = agent.get_metrics()
            
            cars_data.append({
                'id': car_id,
                'name': agent.name,
                'color': agent.color,
                'position': position,
                'speed': car.speed,
                'throttle': car.throttle,
                'brake': car.brake,
                'steering': car.steering,
                'gear': car.gear,
                'rpm': car.rpm,
                'lap_count': car.lap_count,
                'lap_time': agent.current_lap_time,
                'best_lap': agent.best_lap_time,
                'track_position': car.track_position,
                
                # RL metrics
                'rl_metrics': metrics,
                'current_reward': agent.current_reward
            })
        
        # Sort by track position for race order
        cars_data.sort(key=lambda x: (x['lap_count'], x['track_position']), reverse=True)
        for i, car in enumerate(cars_data):
            car['race_position'] = i + 1
        
        return {
            'simulation_active': self.simulation_active,
            'training_active': self.training_active,
            'simulation_time': simulation_time,
            'cars': cars_data,
            'track': self.get_track_data(),
            'episode_data': self.episode_data[-100:],  # Last 100 data points
            'learning_summary': self.get_learning_summary(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_track_data(self):
        """Get track data for 3D rendering"""
        return [
            {
                'x': wp['x'], 'y': wp['y'], 'z': wp['z'],
                'heading': wp['heading'], 'banking': wp['banking'],
                'speed_limit': wp['speed_limit'], 'checkpoint': wp['checkpoint'],
                'start_finish': wp['start_finish']
            }
            for wp in self.track.waypoints[::2]  # Every other waypoint for performance
        ]
    
    def get_learning_summary(self):
        """Get RL learning summary"""
        if not self.agents:
            return {}
        
        summary = {}
        for agent_id, agent in self.agents.items():
            metrics = agent.get_metrics()
            
            summary[agent_id] = {
                'name': agent.name,
                'episodes_completed': agent.episode,
                'total_q_states': len(agent.q_table),
                'exploration_rate': agent.exploration_rate,
                'avg_reward': metrics['avg_reward'],
                'learning_progress': min(1.0, agent.episode / 50.0),  # Progress to 50 episodes
                'performance_trend': 'improving' if len(agent.episode_rewards) > 10 and 
                    np.mean(list(agent.episode_rewards)[-5:]) > np.mean(list(agent.episode_rewards)[:5]) else 'stable'
            }
        
        return summary

# Global system
rl_system = CompleteRLSystem()

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
app = FastAPI(title="Complete TrackMania RL + 3D System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time complete data"""
    await manager.connect(websocket)
    try:
        while True:
            data = rl_system.get_complete_data()
            await websocket.send_json(data)
            await asyncio.sleep(0.05)  # 20 FPS
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/simulation/start")
async def start_simulation():
    """Start complete RL + 3D simulation"""
    success = rl_system.start_simulation()
    return {"status": "started" if success else "already_running"}

@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop simulation"""
    rl_system.stop_simulation()
    return {"status": "stopped"}

@app.get("/api/status")
async def get_status():
    """Get complete system status"""
    return rl_system.get_complete_data()

@app.get("/")
async def root():
    """Complete RL + 3D visualization interface"""
    return HTMLResponse(open('/Users/mac/Desktop/new/trackmania-RL/complete_3d_interface.html').read())

if __name__ == "__main__":
    logger.info("Starting Complete TrackMania RL + 3D System...")
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="info")