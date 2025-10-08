#!/usr/bin/env python3
"""
Production-Level TrackMania RL System
Enhanced 3D graphics, realistic physics, and advanced RL algorithms
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

class ProductionRLAgent:
    """Production-level RL Agent with advanced algorithms"""
    
    def __init__(self, agent_id: str, name: str, color: str, strategy: str = "balanced"):
        self.agent_id = agent_id
        self.name = name
        self.color = color
        self.strategy = strategy
        
        # Advanced Q-Learning with Double DQN concepts
        self.q_table = {}
        self.target_q_table = {}
        self.learning_rate = 0.001 if strategy == "cautious" else 0.003 if strategy == "balanced" else 0.005
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.995
        
        # Advanced metrics
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=200)
        self.q_values_history = deque(maxlen=500)
        self.policy_losses = deque(maxlen=200)
        self.value_estimates = deque(maxlen=200)
        self.training_time = 0.0
        
        # Racing performance metrics
        self.lap_times = deque(maxlen=50)
        self.sector_times = deque(maxlen=200)
        self.overtakes = 0
        self.crashes = 0
        self.fuel_efficiency = 100.0
        
        # Real racing state
        self.current_reward = 0.0
        self.best_lap_time = float('inf')
        self.current_lap_time = 0.0
        self.track_position = random.uniform(0, 0.1)
        self.speed = 0.0
        self.lateral_offset = random.uniform(-2, 2)
        
        # Car control states
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        self.gear = 1
        self.rpm = 1000
        self.tire_temperature = 80.0
        self.fuel_level = 100.0
        
    def get_advanced_state(self, lidar_data: List[float], track_info: Dict, weather: Dict):
        """Advanced state representation with real sensor data"""
        # Discretize LIDAR readings (19 beams)
        lidar_buckets = []
        for reading in lidar_data:
            bucket = min(9, int(reading * 10))  # 0-9 distance buckets
            lidar_buckets.append(bucket)
        
        # Track state
        position_bucket = int(self.track_position * 50) % 50
        speed_bucket = min(9, int(self.speed / 30))
        
        # Environmental factors
        weather_bucket = int(weather.get('grip_factor', 1.0) * 3)
        tire_temp_bucket = min(4, int((self.tire_temperature - 60) / 20))
        fuel_bucket = min(4, int(self.fuel_level / 25))
        
        # Create compound state
        state_tuple = tuple(lidar_buckets[:5] + [position_bucket, speed_bucket, weather_bucket, tire_temp_bucket, fuel_bucket])
        return state_tuple
    
    def get_action(self, state, track_info: Dict):
        """Advanced action selection with strategy-specific behavior"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]  # [throttle, brake, steering]
            self.target_q_table[state] = [0.0, 0.0, 0.0]
        
        if random.random() < self.exploration_rate:
            # Strategy-specific exploration
            if self.strategy == "aggressive":
                throttle = random.uniform(0.6, 1.0)
                brake = random.uniform(0.0, 0.3)
                steering = random.uniform(-1.0, 1.0)
            elif self.strategy == "cautious":
                throttle = random.uniform(0.2, 0.7)
                brake = random.uniform(0.0, 0.6)
                steering = random.uniform(-0.7, 0.7)
            else:  # balanced
                throttle = random.uniform(0.3, 0.9)
                brake = random.uniform(0.0, 0.5)
                steering = random.uniform(-0.8, 0.8)
            
            self.learning_decision = "exploring"
        else:
            # Use Q-values with strategy modification
            q_values = self.q_table[state]
            self.q_values_history.append(max(q_values))
            
            # Strategy-specific exploitation
            base_throttle = 0.6 + q_values[0] * 0.4
            base_brake = max(0.0, q_values[1])
            base_steering = q_values[2]
            
            if self.strategy == "aggressive":
                throttle = min(1.0, base_throttle * 1.2)
                brake = base_brake * 0.8
                steering = base_steering * 1.1
            elif self.strategy == "cautious":
                throttle = base_throttle * 0.8
                brake = min(1.0, base_brake * 1.3)
                steering = base_steering * 0.9
            else:  # balanced
                throttle = base_throttle
                brake = base_brake
                steering = base_steering
            
            # Ensure valid ranges
            throttle = max(0.0, min(1.0, throttle))
            brake = max(0.0, min(1.0, brake))
            steering = max(-1.0, min(1.0, steering))
            
            self.learning_decision = "exploiting"
        
        # Update control states
        self.throttle = throttle
        self.brake = brake
        self.steering = steering
        
        return [throttle, brake, steering]
    
    def update_q_learning(self, state, action, reward, next_state):
        """Advanced Q-learning with Double DQN concepts"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
            self.target_q_table[state] = [0.0, 0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0, 0.0]
            self.target_q_table[next_state] = [0.0, 0.0, 0.0]
        
        # Double DQN update
        current_q = self.q_table[state]
        target_q = self.target_q_table[next_state]
        max_target_q = max(target_q)
        
        # Calculate TD error and update
        total_policy_loss = 0.0
        for i in range(3):
            old_q = current_q[i]
            td_error = reward + self.discount_factor * max_target_q - current_q[i]
            current_q[i] += self.learning_rate * td_error
            
            policy_loss = abs(current_q[i] - old_q)
            total_policy_loss += policy_loss
        
        # Store learning metrics
        self.policy_losses.append(total_policy_loss / 3.0)
        self.value_estimates.append(max(current_q))
        
        # Update target network occasionally
        if self.total_steps % 100 == 0:
            self.target_q_table[state] = current_q.copy()
        
        # Decay exploration
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        self.total_steps += 1
        self.current_reward = reward
    
    def get_comprehensive_metrics(self):
        """Get all production-level metrics"""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_q_value = np.mean(list(self.q_values_history)[-50:]) if self.q_values_history else 0.0
        avg_policy_loss = np.mean(list(self.policy_losses)[-30:]) if self.policy_losses else 0.0
        avg_value_estimate = np.mean(list(self.value_estimates)[-30:]) if self.value_estimates else 0.0
        avg_lap_time = np.mean(list(self.lap_times)[-10:]) if self.lap_times else 0.0
        
        learning_progress = min(100, (self.episode / 100.0) * 100)
        convergence_rate = (1.0 - self.exploration_rate) * 100
        consistency = 100 - (np.std(list(self.episode_rewards)[-20:]) if len(self.episode_rewards) > 20 else 50)
        
        return {
            # Core RL metrics
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
            
            # Performance metrics
            "learning_progress": round(learning_progress, 1),
            "convergence_rate": round(convergence_rate, 1),
            "consistency_score": round(consistency, 1),
            
            # Racing metrics
            "avg_lap_time": round(avg_lap_time, 2),
            "best_lap_time": round(self.best_lap_time, 2) if self.best_lap_time != float('inf') else 0.0,
            "fuel_efficiency": round(self.fuel_efficiency, 1),
            "tire_temperature": round(self.tire_temperature, 1),
            "overtakes": self.overtakes,
            "crashes": self.crashes,
            
            # Car state
            "throttle": round(self.throttle, 3),
            "brake": round(self.brake, 3),
            "steering": round(self.steering, 3),
            "gear": self.gear,
            "rpm": int(self.rpm),
            "speed": round(self.speed, 1),
            
            # Decision making
            "learning_decision": getattr(self, 'learning_decision', 'exploring'),
            "strategy": self.strategy,
            "training_time": self.training_time,
            
            # Historical data for charts
            "reward_history": list(self.episode_rewards)[-100:],
            "q_value_history": list(self.q_values_history)[-100:],
            "policy_loss_history": list(self.policy_losses)[-100:],
            "lap_time_history": list(self.lap_times)[-50:]
        }

class AdvancedTrack3D:
    """Production-level racing circuit with realistic features"""
    
    def __init__(self):
        self.waypoints = []
        self.elevation_map = {}
        self.banking_map = {}
        self.surface_grip = {}
        self.sector_markers = []
        self.generate_professional_circuit()
    
    def generate_professional_circuit(self):
        """Generate Monaco-style street circuit with realistic features"""
        sections = [
            # Section 1: Start/Finish straight
            {"type": "straight", "length": 400, "width": 15, "banking": 0, "elevation": 0, "grip": 1.0},
            # Section 2: Turn 1 (tight hairpin)
            {"type": "hairpin", "radius": 25, "angle": 180, "banking": -5, "elevation": 10, "grip": 0.9},
            # Section 3: Climbing section
            {"type": "straight", "length": 300, "width": 12, "banking": 2, "elevation": 30, "grip": 1.0},
            # Section 4: Chicane complex
            {"type": "chicane", "radius": 40, "width": 10, "banking": 0, "elevation": 35, "grip": 0.95},
            # Section 5: Fast sweeper
            {"type": "sweeper", "radius": 120, "angle": 90, "banking": 8, "elevation": 25, "grip": 1.0},
            # Section 6: Tunnel section
            {"type": "straight", "length": 200, "width": 11, "banking": 0, "elevation": 15, "grip": 0.98},
            # Section 7: Downhill turns
            {"type": "downhill_turns", "radius": 60, "elevation": -20, "banking": -3, "grip": 0.92},
            # Section 8: Final sector
            {"type": "fast_curves", "radius": 80, "banking": 5, "elevation": 0, "grip": 1.0}
        ]
        
        point_index = 0
        for section_idx, section in enumerate(sections):
            points = self.generate_section_points(section, section_idx, point_index)
            self.waypoints.extend(points)
            point_index += len(points)
        
        # Add sector markers (4 sectors)
        total_points = len(self.waypoints)
        for i in range(4):
            marker_point = int(i * total_points / 4)
            self.sector_markers.append(marker_point)
        
        logger.info(f"Generated professional circuit with {len(self.waypoints)} waypoints")
    
    def generate_section_points(self, section: Dict, section_idx: int, start_index: int):
        """Generate waypoints for a track section"""
        points = []
        
        if section["type"] == "straight":
            num_points = max(20, int(section["length"] / 20))
            for i in range(num_points):
                progress = i / num_points
                x = start_index * 10 + progress * section["length"]
                y = 0
                z = section["elevation"]
                
                points.append(self.create_waypoint(
                    x, y, z, section["banking"], section["grip"], 
                    section_idx, i == 0, 250 - section_idx * 20
                ))
        
        elif section["type"] == "hairpin":
            num_points = 30
            for i in range(num_points):
                progress = i / num_points
                angle = progress * math.radians(section["angle"])
                radius = section["radius"]
                
                x = start_index * 10 + radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = section["elevation"] + progress * 5
                
                points.append(self.create_waypoint(
                    x, y, z, section["banking"], section["grip"],
                    section_idx, i == 0, 60 + progress * 40
                ))
        
        elif section["type"] == "chicane":
            num_points = 25
            for i in range(num_points):
                progress = i / num_points
                x = start_index * 10 + progress * 150
                y = 30 * math.sin(progress * 4 * math.pi)  # S-curves
                z = section["elevation"]
                
                points.append(self.create_waypoint(
                    x, y, z, section["banking"], section["grip"],
                    section_idx, i == 0, 120 - abs(y) * 2
                ))
        
        elif section["type"] == "sweeper":
            num_points = 20
            for i in range(num_points):
                progress = i / num_points
                angle = progress * math.radians(section["angle"])
                radius = section["radius"]
                
                x = start_index * 10 + radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = section["elevation"]
                
                points.append(self.create_waypoint(
                    x, y, z, section["banking"], section["grip"],
                    section_idx, i == 0, 180 + progress * 40
                ))
        
        return points
    
    def create_waypoint(self, x: float, y: float, z: float, banking: float, 
                       grip: float, sector: int, is_checkpoint: bool, speed_limit: float):
        """Create a detailed waypoint with all racing information"""
        return {
            'x': x, 'y': y, 'z': z,
            'banking': banking,
            'grip_factor': grip,
            'speed_limit': speed_limit,
            'sector': sector % 4,
            'checkpoint': is_checkpoint,
            'track_width': 12.0,
            'racing_line_offset': -banking * 0.3,
            'braking_zone': speed_limit < 100,
            'elevation_change': z
        }
    
    def get_waypoint(self, position: float):
        """Get waypoint at normalized position"""
        index = int(position * len(self.waypoints)) % len(self.waypoints)
        return self.waypoints[index]
    
    def simulate_lidar(self, car_position: Dict, heading: float, num_beams: int = 19):
        """Simulate realistic LIDAR sensor with noise"""
        lidar_readings = []
        max_range = 100.0  # meters
        
        for i in range(num_beams):
            # Calculate beam angle
            beam_angle = heading + (i - num_beams//2) * (math.pi / (num_beams - 1))
            
            # Cast ray and find distance to track boundaries
            distance = self.cast_lidar_ray(car_position, beam_angle, max_range)
            
            # Add realistic sensor noise
            noise = random.gauss(0, 0.02)  # 2cm standard deviation
            distance_with_noise = max(0.1, distance + noise)
            
            # Normalize to 0-1 range
            normalized_distance = min(1.0, distance_with_noise / max_range)
            lidar_readings.append(normalized_distance)
        
        return lidar_readings
    
    def cast_lidar_ray(self, position: Dict, angle: float, max_range: float):
        """Cast a single LIDAR ray and calculate distance to obstacle"""
        # Simplified ray casting - in production this would be more sophisticated
        step_size = 2.0
        current_distance = 0.0
        
        while current_distance < max_range:
            # Calculate ray position
            ray_x = position['x'] + current_distance * math.cos(angle)
            ray_y = position['y'] + current_distance * math.sin(angle)
            
            # Check if ray hits track boundary (simplified)
            if self.is_off_track(ray_x, ray_y):
                return current_distance
            
            current_distance += step_size
        
        return max_range
    
    def is_off_track(self, x: float, y: float):
        """Check if position is off the racing track"""
        # Simplified track boundary check
        # In production, this would use proper track mesh collision
        track_width = 12.0
        return abs(y) > track_width
    
    def get_weather_conditions(self):
        """Simulate dynamic weather conditions"""
        conditions = random.choice([
            {"condition": "sunny", "grip_factor": 1.0, "visibility": 1.0, "description": "☀️ Perfect"},
            {"condition": "cloudy", "grip_factor": 0.98, "visibility": 0.95, "description": "☁️ Overcast"},
            {"condition": "light_rain", "grip_factor": 0.85, "visibility": 0.8, "description": "🌦️ Light Rain"},
            {"condition": "wet", "grip_factor": 0.7, "visibility": 0.7, "description": "🌧️ Wet Track"}
        ])
        return conditions

class ProductionRacingCar:
    """Advanced racing car with realistic physics and RL integration"""
    
    def __init__(self, agent: ProductionRLAgent, track: AdvancedTrack3D):
        self.agent = agent
        self.track = track
        
        # Advanced physics parameters
        self.mass = 750  # kg (F1 car mass)
        self.max_power = 800  # HP
        self.max_speed = 350  # km/h
        self.drag_coefficient = 0.9
        self.downforce_coefficient = 3.0
        
        # Tire physics
        self.tire_grip = 1.0
        self.tire_wear = 0.0
        self.tire_temperature = 80.0  # °C
        self.optimal_tire_temp = 100.0
        
        # Engine and drivetrain
        self.fuel_consumption_rate = 0.1  # per second at full throttle
        self.gear_ratios = [0, 2.8, 2.1, 1.7, 1.4, 1.2, 1.0]  # 6-speed gearbox
        
        # Racing state
        self.lap_start_time = time.time()
        self.last_sector = -1
        self.sector_start_times = [0.0, 0.0, 0.0, 0.0]
        
        # Environmental factors
        self.weather = self.track.get_weather_conditions()
        
    def update_advanced_physics(self, dt: float = 0.02):
        """Update car with advanced racing physics"""
        old_position = self.agent.track_position
        waypoint = self.track.get_waypoint(self.agent.track_position)
        
        # Get LIDAR data for state
        car_3d_pos = self.get_world_position()
        lidar_data = self.track.simulate_lidar(car_3d_pos, waypoint.get('heading', 0))
        
        # Get current state and action from RL agent
        current_state = self.agent.get_advanced_state(lidar_data, waypoint, self.weather)
        action = self.agent.get_action(current_state, waypoint)
        
        # Apply advanced physics calculations
        self.apply_engine_physics(dt)
        self.apply_aerodynamics(dt)
        self.apply_tire_physics(dt, waypoint)
        self.update_position_and_rotation(dt, waypoint)
        self.update_racing_metrics(dt, waypoint)
        
        # Calculate sophisticated reward
        reward = self.calculate_advanced_reward(old_position, waypoint)
        
        # Update RL learning
        if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            self.agent.update_q_learning(self.last_state, self.last_action, reward, current_state)
        
        self.last_state = current_state
        self.last_action = action
        
        return self.get_world_position()
    
    def apply_engine_physics(self, dt: float):
        """Realistic engine and drivetrain physics"""
        # Calculate engine power based on RPM curve
        optimal_rpm = 6000 + self.agent.gear * 500
        rpm_efficiency = 1.0 - abs(self.agent.rpm - optimal_rpm) / optimal_rpm * 0.3
        
        # Power delivery
        engine_power = self.agent.throttle * self.max_power * rpm_efficiency
        engine_force = engine_power / max(1, self.agent.speed) * 10  # Convert to force
        
        # Braking force
        max_braking_force = 15000  # N
        braking_force = self.agent.brake * max_braking_force
        
        # Calculate net force
        air_resistance = 0.5 * self.drag_coefficient * (self.agent.speed ** 2) * 0.01
        rolling_resistance = 200  # N
        
        net_force = engine_force - braking_force - air_resistance - rolling_resistance
        acceleration = net_force / self.mass
        
        # Update speed
        self.agent.speed += acceleration * dt * 3.6  # Convert to km/h
        self.agent.speed = max(0, min(self.max_speed, self.agent.speed))
        
        # Update RPM based on speed and gear
        if self.agent.gear > 0:
            gear_ratio = self.gear_ratios[self.agent.gear]
            self.agent.rpm = max(800, min(8000, 1000 + self.agent.speed * gear_ratio * 20))
        
        # Auto-shift gears
        if self.agent.rpm > 7500 and self.agent.gear < 6:
            self.agent.gear += 1
        elif self.agent.rpm < 2000 and self.agent.gear > 1:
            self.agent.gear -= 1
    
    def apply_aerodynamics(self, dt: float):
        """Advanced aerodynamic effects"""
        speed_ms = self.agent.speed / 3.6
        
        # Downforce increases grip at high speeds
        downforce = self.downforce_coefficient * (speed_ms ** 2) * 0.01
        self.tire_grip = min(1.5, 1.0 + downforce * 0.001)
        
        # Drag affects fuel consumption
        drag_effect = (speed_ms ** 2) * 0.0001
        self.agent.fuel_level = max(0, self.agent.fuel_level - drag_effect * dt)
    
    def apply_tire_physics(self, dt: float, waypoint: Dict):
        """Realistic tire physics and heating"""
        # Tire temperature effects
        speed_heating = abs(self.agent.speed) * 0.01
        braking_heating = self.agent.brake * 5.0
        steering_heating = abs(self.agent.steering) * 2.0
        
        self.tire_temperature += (speed_heating + braking_heating + steering_heating) * dt
        self.tire_temperature = max(60, min(140, self.tire_temperature))
        
        # Tire performance based on temperature
        temp_diff = abs(self.tire_temperature - self.optimal_tire_temp)
        tire_performance = max(0.6, 1.0 - temp_diff * 0.01)
        
        # Grip calculation
        base_grip = waypoint.get('grip_factor', 1.0)
        weather_grip = self.weather.get('grip_factor', 1.0)
        self.tire_grip = base_grip * weather_grip * tire_performance
        
        # Tire wear
        wear_rate = (abs(self.agent.steering) + self.agent.brake + self.agent.throttle) * 0.001
        self.tire_wear = min(100, self.tire_wear + wear_rate * dt)
        
        # Update agent's tire temperature
        self.agent.tire_temperature = self.tire_temperature
    
    def update_position_and_rotation(self, dt: float, waypoint: Dict):
        """Update car position with realistic handling"""
        # Calculate lateral forces
        max_lateral_g = 4.0 * self.tire_grip  # Max lateral acceleration
        steering_force = self.agent.steering * max_lateral_g * self.agent.speed * 0.01
        
        # Update lateral offset with realistic physics
        banking_effect = waypoint.get('banking', 0) * 0.02
        self.agent.lateral_offset += (steering_force + banking_effect) * dt
        
        # Track limits
        track_width = waypoint.get('track_width', 12.0)
        if abs(self.agent.lateral_offset) > track_width / 2:
            self.agent.crashes += 1
            self.agent.lateral_offset = max(-track_width/2, min(track_width/2, self.agent.lateral_offset))
            self.agent.speed *= 0.5  # Crash penalty
        
        # Update track position
        track_length = 4500  # meters
        speed_ms = self.agent.speed / 3.6
        position_delta = (speed_ms * dt) / track_length
        self.agent.track_position = (self.agent.track_position + position_delta) % 1.0
    
    def update_racing_metrics(self, dt: float, waypoint: Dict):
        """Update racing performance metrics"""
        # Sector timing
        current_sector = waypoint.get('sector', 0)
        if current_sector != self.last_sector and self.last_sector != -1:
            sector_time = time.time() - self.sector_start_times[self.last_sector]
            self.agent.sector_times.append(sector_time)
            self.sector_start_times[current_sector] = time.time()
        
        if self.last_sector == -1:
            self.sector_start_times[current_sector] = time.time()
        
        self.last_sector = current_sector
        
        # Lap completion
        if self.agent.track_position > 0.95 and hasattr(self, 'last_position') and self.last_position < 0.05:
            lap_time = time.time() - self.lap_start_time
            self.agent.lap_times.append(lap_time)
            if lap_time < self.agent.best_lap_time:
                self.agent.best_lap_time = lap_time
            
            self.lap_start_time = time.time()
            self.agent.episode += 1
            self.agent.episode_rewards.append(self.agent.current_reward)
        
        self.last_position = self.agent.track_position
        
        # Fuel efficiency calculation
        fuel_used = self.agent.throttle * self.fuel_consumption_rate * dt
        if fuel_used > 0:
            distance_per_fuel = (self.agent.speed / 3.6 * dt) / fuel_used
            self.agent.fuel_efficiency = min(100, distance_per_fuel * 0.1)
    
    def calculate_advanced_reward(self, old_position: float, waypoint: Dict):
        """Advanced reward function for racing performance"""
        # Progress reward (most important)
        progress_delta = self.agent.track_position - old_position
        if progress_delta < -0.5:  # Lap completion
            progress_delta += 1.0
        progress_reward = progress_delta * 2000
        
        # Speed efficiency reward
        target_speed = waypoint.get('speed_limit', 200)
        speed_ratio = min(1.0, self.agent.speed / target_speed)
        speed_reward = speed_ratio * 100
        
        # Racing line reward
        ideal_offset = waypoint.get('racing_line_offset', 0)
        line_error = abs(self.agent.lateral_offset - ideal_offset)
        line_reward = max(0, 50 - line_error * 3)
        
        # Tire management reward
        tire_temp_penalty = abs(self.tire_temperature - self.optimal_tire_temp) * 0.5
        tire_wear_penalty = self.tire_wear * 0.2
        
        # Fuel efficiency reward
        fuel_reward = self.agent.fuel_efficiency * 0.3
        
        # Smoothness reward
        steering_smoothness = max(0, 20 - abs(self.agent.steering) * 15)
        braking_smoothness = max(0, 15 - self.agent.brake * 10)
        
        # Weather adaptation reward
        weather_grip = self.weather.get('grip_factor', 1.0)
        adaptation_reward = 20 if self.agent.speed < target_speed * weather_grip else 0
        
        # Total reward calculation
        total_reward = (
            progress_reward * 0.35 +
            speed_reward * 0.25 +
            line_reward * 0.15 +
            fuel_reward * 0.08 +
            steering_smoothness * 0.07 +
            braking_smoothness * 0.05 +
            adaptation_reward * 0.05 -
            tire_temp_penalty -
            tire_wear_penalty
        )
        
        return max(-100, min(300, total_reward))
    
    def get_world_position(self):
        """Get detailed 3D world position for rendering"""
        waypoint = self.track.get_waypoint(self.agent.track_position)
        
        # Calculate position with lateral offset
        heading = waypoint.get('heading', 0)
        offset_x = self.agent.lateral_offset * math.cos(heading + math.pi/2)
        offset_y = self.agent.lateral_offset * math.sin(heading + math.pi/2)
        
        return {
            'x': waypoint['x'] + offset_x,
            'y': waypoint['y'] + offset_y,
            'z': waypoint['z'] + 1.5,  # Car height
            'heading': heading + self.agent.steering * 0.4,
            'banking': waypoint.get('banking', 0),
            'track_position': self.agent.track_position,
            'elevation': waypoint.get('z', 0)
        }

class ProductionRLSystem:
    """Production-level RL Racing System"""
    
    def __init__(self):
        self.track = AdvancedTrack3D()
        self.agents = {}
        self.cars = {}
        self.simulation_active = False
        self.start_time = None
        self.training_start_time = None
        self.episode_data = deque(maxlen=200)
        self.weather_update_timer = 0.0
        
    def initialize_production_system(self):
        """Initialize production-level agents and cars"""
        agent_configs = [
            {"id": "prod_balanced", "name": "🏎️ Pro Balanced", "color": "#3498db", "strategy": "balanced"},
            {"id": "prod_aggressive", "name": "🔥 Pro Aggressive", "color": "#e74c3c", "strategy": "aggressive"},
            {"id": "prod_cautious", "name": "🛡️ Pro Cautious", "color": "#2ecc71", "strategy": "cautious"},
            {"id": "prod_adaptive", "name": "🧠 Pro Adaptive", "color": "#9b59b6", "strategy": "balanced"}
        ]
        
        for config in agent_configs:
            # Create advanced RL agent
            agent = ProductionRLAgent(
                config["id"], 
                config["name"], 
                config["color"],
                config["strategy"]
            )
            self.agents[config["id"]] = agent
            
            # Create advanced racing car
            car = ProductionRacingCar(agent, self.track)
            self.cars[config["id"]] = car
        
        logger.info(f"Initialized {len(self.agents)} production-level RL agents")
    
    def start_production_simulation(self):
        """Start production-level simulation"""
        if self.simulation_active:
            return False
        
        self.simulation_active = True
        self.start_time = time.time()
        self.training_start_time = time.time()
        
        if not self.agents:
            self.initialize_production_system()
        
        # Start advanced simulation loop
        asyncio.create_task(self.production_simulation_loop())
        logger.info("Production RL simulation started")
        return True
    
    def stop_production_simulation(self):
        """Stop simulation"""
        self.simulation_active = False
        logger.info("Production simulation stopped")
    
    async def production_simulation_loop(self):
        """Advanced simulation loop with realistic timing"""
        while self.simulation_active:
            dt = 0.02  # 50 FPS for smooth physics
            
            # Update weather conditions periodically
            self.weather_update_timer += dt
            if self.weather_update_timer > 30.0:  # Every 30 seconds
                self.update_weather_conditions()
                self.weather_update_timer = 0.0
            
            # Update all cars with advanced physics
            for car in self.cars.values():
                car.update_advanced_physics(dt)
            
            # Store data every 0.5 seconds
            if int(time.time() * 2) % 1 == 0:
                self.store_production_data()
            
            await asyncio.sleep(dt)
    
    def update_weather_conditions(self):
        """Update weather conditions for all cars"""
        new_weather = self.track.get_weather_conditions()
        for car in self.cars.values():
            car.weather = new_weather
        logger.info(f"Weather updated: {new_weather['description']}")
    
    def store_production_data(self):
        """Store comprehensive production data"""
        current_time = time.time() - self.start_time if self.start_time else 0
        
        episode_data = {
            "timestamp": current_time,
            "weather": self.cars[list(self.cars.keys())[0]].weather if self.cars else {},
            "agents": {}
        }
        
        for agent_id, agent in self.agents.items():
            # Update training time
            if self.training_start_time:
                agent.training_time = current_time
            
            metrics = agent.get_comprehensive_metrics()
            car = self.cars[agent_id]
            position = car.get_world_position()
            
            episode_data["agents"][agent_id] = {
                **metrics,
                "position": position,
                "tire_grip": car.tire_grip,
                "tire_wear": car.tire_wear,
                "weather_adaptation": car.weather.get('grip_factor', 1.0)
            }
        
        self.episode_data.append(episode_data)
    
    def get_production_data(self):
        """Get complete production-level data"""
        simulation_time = time.time() - self.start_time if self.start_time else 0
        
        production_data = []
        for agent_id, agent in self.agents.items():
            car = self.cars[agent_id]
            position = car.get_world_position()
            metrics = agent.get_comprehensive_metrics()
            
            production_data.append({
                'id': agent_id,
                'name': agent.name,
                'color': agent.color,
                'strategy': agent.strategy,
                
                # Enhanced 3D data
                'position': position,
                'speed': agent.speed,
                'lap_count': len(agent.lap_times),
                'track_position': agent.track_position,
                
                # Advanced RL metrics
                'rl_metrics': metrics,
                'learning_decision': getattr(agent, 'learning_decision', 'exploring'),
                
                # Racing performance
                'tire_temperature': agent.tire_temperature,
                'tire_grip': car.tire_grip,
                'tire_wear': car.tire_wear,
                'fuel_level': agent.fuel_level,
                'fuel_efficiency': agent.fuel_efficiency,
                
                # Car telemetry
                'throttle': agent.throttle,
                'brake': agent.brake,
                'steering': agent.steering,
                'gear': agent.gear,
                'rpm': agent.rpm
            })
        
        # Sort by performance
        production_data.sort(key=lambda x: x['rl_metrics']['avg_reward'], reverse=True)
        for i, data in enumerate(production_data):
            data['race_position'] = i + 1
        
        return {
            'simulation_active': self.simulation_active,
            'simulation_time': simulation_time,
            'cars': production_data,
            'track': self.get_enhanced_track_data(),
            'weather': self.cars[list(self.cars.keys())[0]].weather if self.cars else {},
            'learning_summary': self.get_advanced_learning_summary(),
            'episode_history': list(self.episode_data)[-50:],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_enhanced_track_data(self):
        """Get enhanced track data for 3D rendering"""
        return [
            {
                'x': wp['x'], 'y': wp['y'], 'z': wp['z'],
                'banking': wp['banking'], 'grip_factor': wp['grip_factor'],
                'speed_limit': wp['speed_limit'], 'checkpoint': wp['checkpoint'],
                'braking_zone': wp['braking_zone'], 'sector': wp['sector']
            }
            for wp in self.track.waypoints[::3]  # Every 3rd waypoint for performance
        ]
    
    def get_advanced_learning_summary(self):
        """Get advanced learning analytics"""
        if not self.agents:
            return {}
        
        total_episodes = sum(agent.episode for agent in self.agents.values())
        avg_exploration = np.mean([agent.exploration_rate for agent in self.agents.values()])
        total_q_states = sum(len(agent.q_table) for agent in self.agents.values())
        avg_consistency = np.mean([agent.get_comprehensive_metrics()['consistency_score'] for agent in self.agents.values()])
        
        # Find best performing agent
        best_agent = max(self.agents.values(), key=lambda a: a.get_comprehensive_metrics()['avg_reward'])
        
        return {
            'total_episodes': total_episodes,
            'avg_exploration_rate': round(avg_exploration, 4),
            'total_q_states': total_q_states,
            'avg_consistency': round(avg_consistency, 1),
            'best_performer': best_agent.name,
            'learning_stage': 'expert' if avg_exploration < 0.1 else 'advanced' if avg_exploration < 0.3 else 'learning',
            'system_maturity': min(100, total_episodes * 2),
            'performance_spread': round(np.std([a.get_comprehensive_metrics()['avg_reward'] for a in self.agents.values()]), 2)
        }

# Global production system
production_system = ProductionRLSystem()

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
app = FastAPI(title="Production TrackMania RL System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time production data"""
    await manager.connect(websocket)
    try:
        while True:
            data = production_system.get_production_data()
            await websocket.send_json(data)
            await asyncio.sleep(0.05)  # 20 FPS
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/simulation/start")
async def start_production_simulation():
    """Start production simulation"""
    success = production_system.start_production_simulation()
    return {"status": "started" if success else "already_running"}

@app.post("/api/simulation/stop")
async def stop_production_simulation():
    """Stop simulation"""
    production_system.stop_production_simulation()
    return {"status": "stopped"}

@app.get("/api/status")
async def get_production_status():
    """Get production system status"""
    return production_system.get_production_data()

@app.get("/")
async def root():
    """Production-level RL interface"""
    return HTMLResponse(open('/Users/mac/Desktop/new/trackmania-RL/production_interface.html').read())

if __name__ == "__main__":
    logger.info("Starting Production TrackMania RL System...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")