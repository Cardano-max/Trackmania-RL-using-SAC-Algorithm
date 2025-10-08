# TrackMania RL using SAC Algorithm

A complete implementation of Soft Actor-Critic (SAC) and Q-Learning for autonomous driving in TrackMania. Built from scratch with production-ready Docker infrastructure, real-time 3D visualization, and comprehensive testing.

**Author**: Muhammad Ateeb Taseer
**Implementation**: From mathematical foundations without AI assistance

This repository provides a complete, modular Docker setup for training RL agents in TrackMania using the TMRL framework. It includes both the official TMRL implementation and a standalone SAC trainer for testing without the actual game.

## ✨ New Features

- 🎮 **Two-Container Architecture**: Modular environment and model containers communicating via REST API
- 🏎️ **Real-Time 3D Visualization**: Live race playback with Three.js showing agent decisions and learning progress
- 📈 **Synchronized Metrics Dashboard**: Real-time Q-Learning metrics (exploration rate, Q-states, policy loss, rewards) synchronized with 3D visualization
- 🤖 **Multi-Agent Racing**: Three concurrent agents with different learning strategies (Cautious, Smart, Aggressive)
- 🎯 **LIDAR-Based Observation**: 19-beam LIDAR sensors for realistic environment perception
- 📊 **Advanced Metrics**: Q-value evolution, policy loss tracking, convergence rate, training time, reward history
- 🎨 **Visual Learning Indicators**: Color-coded indicators showing exploration vs exploitation in real-time
- 🐳 **Production-Ready Deployment**: Separate Docker containers for environment, model, and viewer services

## Features

- 🐳 **Fully Dockerized**: Complete setup with server, trainer, and rollout components
- 🧠 **Multiple Training Options**: Official TMRL or standalone SAC implementation
- 📊 **Monitoring**: TensorBoard integration for training visualization
- 🚗 **Mock Environment**: Test training without TrackMania game
- 🔧 **Modular**: Easy to configure and extend
- 🎮 **Live 3D Racing Visualization**: Watch agents learn and race in real-time

## Prerequisites

- Docker and Docker Compose
- For GPU training: NVIDIA Docker runtime (optional)
- For actual game training: TrackMania 2020 with OpenPlanet

## Quick Start

### Option 1: Synchronized RL + 3D Visualization System (Recommended for Demo)

Experience the complete system with real-time Q-Learning and 3D racing visualization:

```bash
# Activate virtual environment
source venv/bin/activate

# Quick start with demo script
./start_demo.sh

# Or manually:
python3 synchronized_rl_3d_system.py
```

Then visit:
- **Main Dashboard**: http://localhost:7001 - Live 3D racing with synchronized RL metrics
- Watch three agents race and learn in real-time with visible learning indicators

### Option 2: Two-Container Architecture

Run the modular environment and model containers separately:

```bash
# Terminal 1 - Start Environment Container
cd environment
python3 environment_server.py  # Port 8080

# Terminal 2 - Start Model Container
cd model
python3 model_server.py  # Port 8081

# Test container communication
curl -X POST http://localhost:8080/api/simulation/start
```

### Option 3: Traditional Docker Setup

```bash
cd docker
docker compose build
docker compose up -d
```

This starts:
- **tmrl-server**: Coordination server (port 5555)
- **tmrl-trainer**: Training service (port 5556)
- **tensorboard**: Monitoring dashboard (port 6006)

### Monitor Training

- **3D Racing Dashboard**: http://localhost:7001 (synchronized system)
- **TensorBoard**: http://localhost:6006 (Docker setup)
- **Logs**: `docker logs tmrl-trainer -f`

### Test Environment (Optional)

To test with mock rollout worker:

```bash
docker compose --profile with-rollout up tmrl-rollout
```

### Connect Real TrackMania Client

On a Windows PC with TrackMania 2020:

```powershell
powershell -ExecutionPolicy Bypass -File .\clients\windows_rollout_client.ps1 -TmrlDataPath "$env:USERPROFILE\TmrlData" -ServerIp "<your-server-ip>" -ServerPort 5555
```

## Configuration

The system uses `/TmrlData/config/config.json` for configuration:

### Basic Security
```json
{
  "PASSWORD": "your-strong-password-here",
  "TLS": false
}
```

### Training Algorithm (SAC)
```json
{
  "ALG": {
    "ALGORITHM": "SAC",
    "LR_ACTOR": 0.0003,
    "LR_CRITIC": 0.0005,
    "GAMMA": 0.995,
    "ALPHA": 0.2
  }
}
```

### Environment Settings
The config includes LIDAR environment setup:
- 19-beam LIDAR observations
- Speed and action history
- Customizable reward function

## Training Modes

### 1. Full TMRL Training
Uses the official TMRL framework with real TrackMania game.

### 2. Standalone SAC Training  
Custom SAC implementation with mock environment for testing:
- No game required
- Immediate feedback
- Algorithm validation

## File Structure

```
trackmania-RL/
├── synchronized_rl_3d_system.py    # Main RL+3D visualization system (port 7001)
├── environment/
│   └── environment_server.py       # Environment container (port 8080)
├── model/
│   └── model_server.py             # Model container (port 8081)
├── viewer/
│   ├── viewer_server.py            # 3D viewer service
│   └── trackmania_template.html    # Three.js racing template
├── docker/
│   ├── docker-compose.yml          # Main orchestration
│   ├── Dockerfile.environment      # Environment container build
│   ├── Dockerfile.model            # Model container build
│   ├── Dockerfile.viewer           # Viewer container build
│   ├── server.Dockerfile           # TMRL server
│   ├── trainer.Dockerfile          # Training service
│   └── rollout.Dockerfile          # Rollout worker
├── scripts/
│   ├── setup_env.py                # Mock environment
│   ├── train_sac.py                # Standalone SAC trainer
│   └── test_environment.py         # Environment tests
├── start_demo.sh                   # Quick demo startup script
├── check_training.py               # Training verification
├── tmrl_templates/                 # Configuration templates
├── clients/                        # Client connection scripts
└── FINAL_VIDEO_SCRIPT_COMPLETE.md  # Complete demo documentation
```

## GPU Support

To enable GPU training, modify the Dockerfiles:

```dockerfile
# Replace CPU PyTorch with GPU version
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

And add GPU access to docker-compose.yml:

```yaml
services:
  tmrl-trainer:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Security Notes

⚠️ **Important**: Default setup uses unencrypted communication for localhost testing.

For production/public networks:
1. Set `"TLS": true` in config.json
2. Generate TLS certificates
3. Use strong passwords
4. Follow TMRL security guidelines

## Monitoring & Debugging

- **Training Progress**: TensorBoard at http://localhost:6006
- **Container Logs**: `docker logs <container-name> -f`
- **Data Volume**: `docker volume inspect trackmania-rl_tmrl_data`
- **Test Environment**: `docker exec -it tmrl-trainer python /app/scripts/test_environment.py`

## Troubleshooting

### Container Issues
```bash
# Check container status
docker compose ps

# View logs
docker logs tmrl-trainer -f

# Rebuild if needed
docker compose build --no-cache
```

### Training Issues
```bash
# Test mock environment
docker exec -it tmrl-trainer python /app/scripts/test_environment.py

# Check configuration
docker exec -it tmrl-trainer cat /TmrlData/config/config.json
```

## Advanced Usage

### Custom Environments
Modify `scripts/setup_env.py` to create custom training environments.

### Algorithm Tuning
Edit hyperparameters in `tmrl_templates/config.trainer.json`.

### Multi-Agent Training
Scale rollout workers by running multiple client connections.

## System Architecture

### Synchronized System (Port 7001)
- **Q-Learning Agents**: Three agents with different exploration strategies
- **Real-Time Metrics**: Episodes, exploration rate, Q-states, policy loss, rewards
- **3D Visualization**: Three.js rendering with color-coded learning indicators
- **WebSocket Communication**: Live updates synchronized between metrics and visualization

### Two-Container Architecture
- **Environment Container (8080)**: TrackMania physics simulation, LIDAR sensors, race recording
- **Model Container (8081)**: Q-Learning and SAC agents, training logic, policy networks
- **REST API Communication**: Modular design for easy deployment and scaling

### Key Metrics Explained
- **Episodes**: Number of complete races finished by each agent
- **Exploration Rate**: Percentage of random actions (100% = pure exploration, 0% = pure exploitation)
- **Q-States**: Size of agent's knowledge base (discrete state representations)
- **Policy Loss**: How much the agent's strategy is changing (lower = more stable)
- **Convergence Rate**: Transition from exploration to exploitation (higher = more learned behavior)

## Demo Materials

Complete video demonstration scripts and client presentation materials are included:
- `FINAL_VIDEO_SCRIPT_COMPLETE.md` - 15-minute video script with metric explanations
- `MESSAGE_TO_HENRIQUE.md` - Professional client summary
- `EMAIL_TO_SEND.txt` - Ready-to-send email template
- `READY_TO_SEND_CHECKLIST.md` - Submission checklist

## Contributing

This setup provides a foundation for TrackMania RL research. Contributions welcome for:
- Additional algorithms (PPO, DDPG, etc.)
- Enhanced environments
- Performance optimizations
- Documentation improvements

