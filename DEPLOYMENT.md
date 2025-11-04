# Production Deployment Guide

This guide covers production deployment of the MoE Ollama endpoint using Podman (Docker-compatible).

## Architecture Overview

```
┌─────────────────────┐
│   MoE Endpoint      │  FastAPI + Router + RAG + Vision + Tools
│   (moe-endpoint)    │  Port: 8000
└──────────┬──────────┘
           │
           │  HTTP/JSON
           ▼
┌─────────────────────┐
│  Ollama Cloud API   │  9 specialized models
│  api.ollama.com     │  (Remote service)
└─────────────────────┘

┌─────────────────────┐
│  PostgreSQL         │  RAG document storage
│  + pgvector         │  Port: 5432
│  (pgvector)         │
└─────────────────────┘
```

## Container Setup (Podman)

### 1. Install Podman

**Windows 11 Home:**
```powershell
winget install RedHat.Podman
# Enable WSL2 backend during installation
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install podman

# RHEL/Fedora
sudo dnf install podman
```

**macOS:**
```bash
brew install podman
podman machine init
podman machine start
```

### 2. Build the Application Container

```bash
# Clone repository
git clone https://github.com/Grumpified-OGGVCT/moe-ollama-endpoint-grumpified.git
cd moe-ollama-endpoint-grumpified

# Build image
podman build -t moe-endpoint:latest -f Containerfile .
```

### 3. Run PostgreSQL with pgvector

```bash
podman run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=moe_user \
  -e POSTGRES_DB=moe_rag \
  -p 5432:5432 \
  --restart unless-stopped \
  ankane/pgvector:latest
```

### 4. Run the MoE Endpoint

```bash
# Create .env file first (see Configuration section)
podman run -d \
  --name moe-endpoint \
  -e OLLAMA_API_KEY=$OLLAMA_API_KEY \
  --env-file .env \
  -p 8000:8000 \
  --restart unless-stopped \
  moe-endpoint:latest
```

### 5. Using Podman Compose

Alternatively, use the provided `podman-compose.yml`:

```bash
# Start all services
podman-compose up -d

# View logs
podman-compose logs -f

# Stop services
podman-compose down
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Ollama Cloud API
OLLAMA_BASE_URL=https://api.ollama.com
OLLAMA_API_KEY=your_ollama_api_key_here

# PostgreSQL / pgvector
POSTGRES_HOST=pgvector
POSTGRES_PORT=5432
POSTGRES_USER=moe_user
POSTGRES_PASSWORD=postgres
POSTGRES_DB=moe_rag

# Application
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

# MoE Routing
TOP_K=2                  # Number of experts per request
THINKING_MODE=high       # Default think level (low/medium/high)

# Backup/Failover Thresholds
MAX_LATENCY_MS=2000      # >2s triggers fallback
MAX_RETRIES=3            # Maximum retry attempts
CIRCUIT_BREAKER_THRESHOLD=3

# Model Configuration (Optional - uses defaults if not set)
REASONING_MODEL=deepseek-v3.1:671b-cloud
FALLBACK_MODEL=gpt-oss:20b-cloud
ENTERPRISE_MODEL=gpt-oss:120b-cloud
MATH_TOOL_MODEL=kimi-k2:1t-cloud
CODE_MODEL=qwen3-coder:480b-cloud
AGGREGATOR_MODEL=glm-4.6:cloud
COST_CODE_MODEL=minimax-m2:cloud
VISION_MODEL=qwen3-vl:235b-cloud
VISION_THINKING_MODEL=qwen3-vl:235b-instruct-cloud
DEFAULT_MODEL=gpt-oss:20b-cloud

# RAG Configuration
EMBEDDING_MODEL=nomic-embed-text
VECTOR_DIMENSION=768
TOP_K_RESULTS=5

# Monitoring (Optional)
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Security Considerations

1. **API Keys**: Never commit `.env` to version control
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Secrets Management**: For production, use:
   - Kubernetes Secrets
   - HashiCorp Vault
   - Cloud provider secret managers (AWS Secrets Manager, Azure Key Vault)

3. **Network Security**:
   ```bash
   # Create dedicated network for containers
   podman network create moe-network
   
   # Run containers on isolated network
   podman run --network moe-network ...
   ```

4. **TLS/HTTPS**: Use reverse proxy (nginx/traefik) for SSL termination

## CI/CD Pipeline

### GitHub Actions with Podman

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      
      - name: Run tests
        run: pytest tests/ -v

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Podman
        run: |
          sudo apt-get update
          sudo apt-get install -y podman
      
      - name: Build image
        run: |
          podman build -t ghcr.io/${{ github.repository }}:${{ github.sha }} .
          podman tag ghcr.io/${{ github.repository }}:${{ github.sha }} \
                     ghcr.io/${{ github.repository }}:latest
      
      - name: Security scan
        uses: aquasecurity/trivy-action@0.28.0
        with:
          image-ref: ghcr.io/${{ github.repository }}:${{ github.sha }}
          format: table
          exit-code: 1
          ignore-unfixed: true
          severity: CRITICAL,HIGH
      
      - name: Login to GitHub Container Registry
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | \
            podman login ghcr.io -u ${{ github.actor }} --password-stdin
      
      - name: Push image
        run: |
          podman push ghcr.io/${{ github.repository }}:${{ github.sha }}
          podman push ghcr.io/${{ github.repository }}:latest
```

## Monitoring and Observability

### Prometheus Metrics

The endpoint exposes metrics at `/metrics`:

```yaml
# Key metrics
moe_request_count_total          # Total requests
moe_expert_invocations_total     # Invocations per model
moe_fallback_count_total         # Fallback activations
moe_request_duration_seconds     # Request latency histogram
moe_circuit_breaker_active       # Circuit breaker status
```

### Grafana Dashboards

Import the provided dashboard (`monitoring/grafana-dashboard.json`):

1. Request throughput by expert
2. Latency percentiles (p50, p95, p99)
3. Fallback/error rates
4. Cost tracking (tokens × model)
5. Circuit breaker activations

### Logging

Structured JSON logging to stdout:

```json
{
  "timestamp": "2025-11-04T14:56:49Z",
  "level": "INFO",
  "message": "Routing to vision+thinking model",
  "model": "qwen3-vl:235b-instruct-cloud",
  "request_id": "abc-123",
  "latency_ms": 1234
}
```

Collect with:
- **Loki** (Grafana stack)
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **CloudWatch Logs** (AWS)

### Error Tracking

Integrate Sentry for error monitoring:

```env
SENTRY_DSN=https://your-sentry-dsn
SENTRY_ENVIRONMENT=production
```

## Operational Checklist

### Pre-Deployment
- [ ] Podman installed and configured
- [ ] `.env` file created with valid `OLLAMA_API_KEY`
- [ ] PostgreSQL/pgvector container running
- [ ] Network connectivity to `api.ollama.com` verified
- [ ] SSL certificates configured (if using HTTPS)

### Initial Deployment
- [ ] Build application container
- [ ] Run health check: `curl http://localhost:8000/health`
- [ ] Verify database connection
- [ ] Load RAG corpus (if applicable)
- [ ] Test each expert model with sample requests

### Monitoring Setup
- [ ] Prometheus scraping `/metrics` endpoint
- [ ] Grafana dashboards configured
- [ ] Alerts configured for critical metrics
- [ ] Error tracking (Sentry) enabled
- [ ] Log aggregation configured

### Security Hardening
- [ ] API keys stored in secrets manager
- [ ] Containers running as non-root user
- [ ] Network isolation configured
- [ ] HTTPS/TLS enabled
- [ ] Rate limiting configured
- [ ] CORS policies set

### Backup and Recovery
- [ ] Database backup schedule configured
- [ ] Backup retention policy defined
- [ ] Disaster recovery plan documented
- [ ] Failover testing performed

## Scaling Considerations

### Horizontal Scaling

Run multiple endpoint instances:

```bash
# Scale to 3 replicas
podman-compose up -d --scale moe-endpoint=3

# Use load balancer (nginx, traefik, HAProxy)
```

### Database Scaling

- **Read replicas**: For RAG queries
- **Connection pooling**: Configure pgBouncer
- **Sharding**: For very large RAG corpora

### Cost Optimization

1. **Cache frequently used queries**
2. **Use cheaper models for simple queries**
3. **Batch similar requests**
4. **Monitor token usage per model**
5. **Set budget alerts in Prometheus**

## Troubleshooting

### Health Check Failing

```bash
# Check container logs
podman logs moe-endpoint

# Verify environment variables
podman exec moe-endpoint env | grep OLLAMA

# Test Ollama API connectivity
curl -H "Authorization: Bearer $OLLAMA_API_KEY" \
     https://api.ollama.com/api/tags
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
podman ps | grep pgvector

# Test connection
PGPASSWORD=postgres psql -h localhost -U moe_user -d moe_rag -c '\dt'

# Check pgvector extension
psql -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

### Model Not Available

```bash
# List available models via Ollama API
curl -H "Authorization: Bearer $OLLAMA_API_KEY" \
     https://api.ollama.com/api/tags

# Check router configuration
podman exec moe-endpoint python -c \
  "from app.services.router import moe_router; print(moe_router.get_model_info())"
```

### High Latency

1. Check backup thresholds: `MAX_LATENCY_MS`
2. Verify network latency to Ollama API
3. Review circuit breaker activations
4. Check if fallback models are being used
5. Monitor Prometheus latency metrics

## Support and Resources

- **Documentation**: [README.md](README.md)
- **Model Verification**: [MODELS.md](MODELS.md)
- **Backup Strategy**: [BACKUP_STRATEGY.md](BACKUP_STRATEGY.md)
- **API Reference**: [API.md](API.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

For issues and feature requests, please use GitHub Issues.
