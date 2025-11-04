#!/bin/bash
# Quick start script for local development

set -e

echo "🚀 Starting MoE Ollama Endpoint..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env and add your OLLAMA_API_KEY before continuing."
    exit 1
fi

# Check if podman-compose is available
if command -v podman-compose &> /dev/null; then
    echo "✓ Using podman-compose"
    COMPOSE_CMD="podman-compose"
elif command -v docker-compose &> /dev/null; then
    echo "✓ Using docker-compose"
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Neither podman-compose nor docker-compose found!"
    echo "Please install one of them to continue."
    exit 1
fi

# Start services
echo ""
echo "Starting services with $COMPOSE_CMD..."
$COMPOSE_CMD up -d

echo ""
echo "✓ Services started successfully!"
echo ""
echo "📊 Service status:"
$COMPOSE_CMD ps

echo ""
echo "🔗 API available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "To view logs: $COMPOSE_CMD logs -f"
echo "To stop: $COMPOSE_CMD down"
