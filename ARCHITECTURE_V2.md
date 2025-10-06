# TrackMania RL - Two-Container Architecture Design

## 🎯 **Requirements from Henrique:**

1. **Modular Design**: Environment container + Model container
2. **Visual Replay**: Graphical race playback for demonstrations
3. **Professional Presentation**: Beautiful visualizations for stakeholders
4. **Future-Proof**: Easy environment swapping

## 🏗️ **Architecture Overview:**

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│     ENVIRONMENT CONTAINER   │    │       MODEL CONTAINER       │
│                             │    │                             │
│  ┌─────────────────────────┐│    │  ┌─────────────────────────┐│
│  │   TrackMania Simulator  ││    │  │     SAC Agent           ││
│  │   - Physics Engine      ││    │  │   - Neural Networks     ││
│  │   - LIDAR Simulation    ││    │  │   - Training Logic      ││
│  │   - Track Management    ││    │  │   - Experience Replay   ││
│  └─────────────────────────┘│    │  └─────────────────────────┘│
│                             │    │                             │
│  ┌─────────────────────────┐│    │  ┌─────────────────────────┐│
│  │  Graphics Renderer      ││    │  │   Action Generator      ││
│  │   - Real-time Racing    ││    │  │   - Policy Execution    ││
│  │   - Multiple Agents     ││    │  │   - State Processing    ││
│  │   - Camera System       ││    │  │   - Reward Calculation  ││
│  └─────────────────────────┘│    │  └─────────────────────────┘│
│                             │    │                             │
│  ┌─────────────────────────┐│    │  ┌─────────────────────────┐│
│  │    Replay System        ││    │  │    Communication       ││
│  │   - Race Recording      ││    │  │   - REST API Client     ││
│  │   - Trajectory Storage  ││    │  │   - Action Streaming    ││
│  │   - Playback Controls   ││    │  │   - State Reception     ││
│  └─────────────────────────┘│    │  └─────────────────────────┘│
│                             │    │                             │
│         Port: 8080          │◄──►│         Port: 8081          │
└─────────────────────────────┘    └─────────────────────────────┘
```

## 🔄 **Communication Protocol:**

### **Model → Environment:**
```json
{
  "agent_id": "sac_agent_1",
  "action": {
    "gas": 0.8,
    "brake": 0.0,
    "steering": 0.2
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### **Environment → Model:**
```json
{
  "agent_id": "sac_agent_1", 
  "observation": {
    "speed": 120.5,
    "lidar": [0.8, 0.9, 0.7, ...],
    "position": {"x": 100.0, "y": 50.0, "z": 0.0},
    "rotation": {"yaw": 0.5, "pitch": 0.0, "roll": 0.0}
  },
  "reward": 15.2,
  "done": false,
  "info": {
    "track_completion": 0.25,
    "lap_time": 45.6
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 🎮 **Environment Container Features:**

### **Core Simulation:**
- Physics-based car dynamics
- Track generation and management
- LIDAR sensor simulation
- Collision detection
- Weather/lighting effects

### **Graphics Rendering:**
- Real-time 3D race visualization
- Multiple camera angles (chase, overhead, cockpit)
- Agent trajectory trails with color coding
- Performance metrics overlay
- Smooth interpolation for cinematic quality

### **Replay System:**
- Race recording in compressed format
- Timestamp-synchronized playback
- Speed controls (0.5x to 4x)
- Agent comparison modes
- Export to video formats

### **API Endpoints:**
```
POST /api/action          # Receive agent actions
GET  /api/state           # Send environment state  
POST /api/reset           # Reset environment
GET  /api/replay/{id}     # Get replay data
POST /api/agents/add      # Add new agent
DELETE /api/agents/{id}   # Remove agent
```

## 🤖 **Model Container Features:**

### **SAC Implementation:**
- Twin Q-networks with target stabilization
- Experience replay buffer (scalable)
- Automatic entropy tuning
- Distributed training support

### **Agent Management:**
- Multiple agent instances
- Independent learning processes
- Shared experience pools
- Performance tracking

### **Communication:**
- Async action streaming
- State processing pipeline
- Reward aggregation
- Training metrics export

## 🎨 **Visual Demonstration Features:**

### **Race Replay:**
- Smooth camera transitions
- Agent trail visualization
- Speed/steering overlays
- Performance comparisons
- Time-lapse capabilities

### **Training Progress:**
- Real-time learning curves
- Episode highlights
- Failure analysis modes
- Improvement tracking

### **Presentation Mode:**
- Professional layouts
- Branded interface
- Export capabilities
- Multiple viewing angles

## 🐳 **Docker Implementation:**

### **Environment Container (trackmania-env):**
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    xvfb x11vnc fluxbox \
    ffmpeg imagemagick
COPY environment/ /app/
EXPOSE 8080 5900
CMD ["python3", "/app/environment_server.py"]
```

### **Model Container (trackmania-model):**
```dockerfile
FROM python:3.10-slim
RUN pip install torch torchvision gymnasium numpy
COPY model/ /app/
EXPOSE 8081
CMD ["python3", "/app/model_server.py"]
```

### **Docker Compose:**
```yaml
version: "3.9"
services:
  trackmania-env:
    build: ./environment
    ports: ["8080:8080", "5900:5900"]
    volumes: ["./data:/data"]
    environment: ["DISPLAY=:0"]
    
  trackmania-model:
    build: ./model  
    ports: ["8081:8081"]
    volumes: ["./data:/data"]
    depends_on: ["trackmania-env"]
    
  replay-viewer:
    build: ./viewer
    ports: ["3000:3000"]
    volumes: ["./data:/data"]
```

## 📊 **Technical Specifications:**

### **Performance Requirements:**
- Environment: 60 FPS rendering, <10ms action response
- Model: <5ms inference time, 1000+ episodes/hour
- Communication: <1ms latency, 100+ actions/second

### **Data Storage:**
- Race replays: Compressed trajectory format
- Training data: HDF5 for efficient access
- Models: PyTorch checkpoint format
- Metrics: Time-series database

### **Scalability:**
- Multiple model containers per environment
- Environment clustering for parallel training
- Cloud deployment ready
- Auto-scaling based on demand

## 🚀 **Implementation Timeline:**

### **Day 1-2: Container Separation**
- Split existing code into two containers
- Implement REST API communication
- Basic environment simulation
- Model inference pipeline

### **Day 3-4: Graphics & Replay**
- 3D rendering system
- Race recording/playback
- Camera system
- Visual effects

### **Day 5-6: Integration & Polish**
- End-to-end testing
- Performance optimization
- Demo preparation
- Documentation

## 🎯 **Success Metrics:**

- ✅ Containers communicate seamlessly
- ✅ Beautiful race replay with multiple agents
- ✅ Professional demo quality
- ✅ Easy environment swapping
- ✅ Impressive stakeholder presentations
- ✅ Foundation for future projects