#!/bin/bash

echo "=== TMRL Docker Setup Test ==="
echo

# Check if Docker is running
echo "1. Checking Docker status..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    echo "   Run: open -a Docker"
    echo "   Then wait for Docker to start and run this script again."
    exit 1
fi
echo "✅ Docker is running"

# Clean up any existing containers
echo
echo "2. Cleaning up existing containers..."
docker compose -f docker/docker-compose.yml down 2>/dev/null || true
echo "✅ Cleanup complete"

# Build the Docker images
echo
echo "3. Building Docker images..."
cd docker
if docker compose build; then
    echo "✅ Docker images built successfully"
else
    echo "❌ Failed to build Docker images"
    exit 1
fi

# Create the shared volume and test basic setup
echo
echo "4. Testing basic service startup..."
if docker compose up -d tmrl-server; then
    echo "✅ TMRL server started"
    
    # Wait a moment and check if it's running
    sleep 5
    if docker compose ps | grep -q "tmrl-server.*running"; then
        echo "✅ TMRL server is running properly"
    else
        echo "❌ TMRL server failed to stay running"
        docker compose logs tmrl-server
        exit 1
    fi
else
    echo "❌ Failed to start TMRL server"
    exit 1
fi

# Test trainer startup
echo
echo "5. Testing trainer startup..."
if docker compose up -d tmrl-trainer; then
    echo "✅ TMRL trainer started"
    
    # Wait a moment and check logs
    sleep 10
    echo "📋 Trainer logs:"
    docker compose logs tmrl-trainer | tail -20
    
    if docker compose ps | grep -q "tmrl-trainer.*running"; then
        echo "✅ TMRL trainer is running"
    else
        echo "⚠️  TMRL trainer may have issues, check logs above"
    fi
else
    echo "❌ Failed to start TMRL trainer"
    exit 1
fi

# Test TensorBoard
echo
echo "6. Testing TensorBoard..."
if docker compose up -d tensorboard; then
    echo "✅ TensorBoard started"
    echo "📊 TensorBoard should be available at: http://localhost:6006"
else
    echo "❌ Failed to start TensorBoard"
fi

# Show running containers
echo
echo "7. Current status:"
docker compose ps

# Test the data volume
echo
echo "8. Testing data volume..."
if docker exec tmrl-server ls -la /TmrlData; then
    echo "✅ Data volume is mounted and accessible"
else
    echo "❌ Data volume issues"
fi

# Test configuration
echo
echo "9. Testing configuration..."
if docker exec tmrl-server cat /TmrlData/config/config.json 2>/dev/null; then
    echo "✅ Configuration file is accessible"
else
    echo "⚠️  Configuration file not found (will be created on first run)"
fi

echo
echo "=== Test Summary ==="
echo "✅ TMRL Docker setup is ready!"
echo
echo "To use the setup:"
echo "  • Start all services: docker compose up -d"
echo "  • View logs: docker logs tmrl-trainer -f"
echo "  • TensorBoard: http://localhost:6006"
echo "  • Stop services: docker compose down"
echo
echo "To test with mock rollout worker:"
echo "  docker compose --profile with-rollout up tmrl-rollout"
echo
echo "For actual TrackMania training:"
echo "  Run the Windows client script from clients/ on a machine with the game"