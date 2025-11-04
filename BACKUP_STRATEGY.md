# Backup and Redundancy Strategy

This document describes the comprehensive backup and failover strategy for the MoE Ollama endpoint to ensure production resilience.

## Backup Chain Strategy

A robust production service must survive **temporary unavailability**, **rate-limit throttling**, or **excessive latency** of any cloud model. The plan defines a *primary → secondary* chain for each expert, plus a *global* fallback guaranteeing the endpoint always returns an answer.

### Backup Chains Table

| Primary Expert (Use-Case) | First Backup (Same Modality) | Second Backup (If Both Fail) | Rationale |
|---------------------------|------------------------------|------------------------------|-----------|
| `deepseek-v3.1:671b-cloud` (complex reasoning) | `gpt-oss:120b-cloud` (deep reasoning, lower latency) | `gpt-oss:20b-cloud` (cheap generic) | Keeps a "reasoning-capable" model alive; if the 671B MoE is down we fall back to smaller but still strong model |
| `gpt-oss:20b-cloud` (fallback) | `gpt-oss:120b-cloud` (if better quality needed) | **none** - already the cheapest baseline | The 20B model is the *built-in cheap safety net*; it only falls back to itself |
| `gpt-oss:120b-cloud` (enterprise) | `deepseek-v3.1:671b-cloud` (alternative chain-of-thought) | `gpt-oss:20b-cloud` | Two high-capacity models can cover each other |
| `kimi-k2:1t-cloud` (math/tool) | `glm-4.6:cloud` (tool-use, reasoning) | `gpt-oss:120b-cloud` (generic math) | If Kimi-K2 is throttled, GLM-4.6 still handles tool calls; 120B model offers decent math support as last resort |
| `qwen3-coder:480b-cloud` (code) | `minimax-m2:cloud` (coding, cheaper) | `gpt-oss:20b-cloud` (basic code snippets) | Guarantees a coding answer even when flagship coder model is offline |
| `glm-4.6:cloud` (aggregator) | `deepseek-v3.1:671b-cloud` (reasoning + synthesis) | `gpt-oss:120b-cloud` | Aggregator must always be reachable; any high-capacity model can play "synthesizer" role in a pinch |
| `minimax-m2:cloud` (cost-coding) | `gpt-oss:20b-cloud` (generic) | **none** - already the cheapest |
| `qwen3-vl:235b-cloud` (vision) | `qwen3-vl:235b-instruct-cloud` (if "thinking" version needed) | **text fallback** - `gpt-oss:20b-cloud` (ignore image, answer based on accompanying text) |
| `qwen3-vl:235b-instruct-cloud` (vision + thinking) | `qwen3-vl:235b-cloud` (plain visual) | **text fallback** - `gpt-oss:20b-cloud` |

## Fallback Trigger Mechanisms

### 1. Latency Monitor
- Each async call records start-time
- If elapsed time > 2s, request is cancelled and marked "slow"
- Automatically triggers fallback to next model in chain

### 2. HTTP Status Handling
- **429 (rate-limit)**: Immediate retry with exponential back-off (max 3 attempts)
- **5xx errors**: Same retry logic
- After max retries, move to next backup model

### 3. Circuit Breaker Pattern
- After 3 consecutive failures, model is placed in *quarantined* set
- Quarantine lasts for remainder of request batch
- Forces router to select next backup automatically
- Prevents cascading failures

### 4. Graceful Degradation
When both vision models are unavailable:
1. Router strips the `images` array from request
2. Logs warning about image content being ignored
3. Routes text-only portion to text model (e.g., `gpt-oss:20b-cloud`)
4. Client still receives a response (albeit without image understanding)

## Implementation Details

All fallback logic is implemented in the **router** (`app/services/router.py`):

```python
# Example usage
model, use_rag = await moe_router.route_request(messages)

# Get backup models for failover
backups = moe_router.get_backup_models(model)
# Returns list of backup models in priority order
```

### Configuration Parameters

Set via environment variables:

```env
# Backup thresholds
MAX_LATENCY_MS=2000      # >2s triggers fallback
MAX_RETRIES=3            # Maximum retry attempts
CIRCUIT_BREAKER_THRESHOLD=3  # Failures before quarantine
```

## Monitoring and Observability

### Metrics to Track
- **Fallback invocations**: Count per primary→backup transition
- **Circuit breaker activations**: When models are quarantined
- **Latency percentiles**: p50, p95, p99 per model
- **Error rates**: By model and error type

### Recommended Alerts
1. **High fallback rate** (>10% of requests) - Investigate primary model health
2. **Circuit breaker active** - Critical: Primary model unavailable
3. **Degraded mode** - Vision requests falling back to text-only
4. **Cost anomaly** - Unexpected shift to expensive backup models

## Testing Recommendations

### Disaster Recovery Tests
Run weekly to validate failover:

1. **Simulate unavailability**: Set `OLLAMA_BASE_URL=https://invalid`
2. **Verify fallback chain**: Confirm requests succeed via backup models
3. **Test vision degradation**: Block vision models, verify text-only fallback
4. **Load test fallback**: Ensure backup models handle primary traffic

### Integration Tests
Include in CI/CD pipeline:
- Test each primary→backup transition
- Verify circuit breaker activation
- Confirm graceful degradation for vision
- Validate retry logic with mock 429/5xx responses

## Future Enhancements

Potential improvements to backup strategy:

1. **Dynamic backup selection**: Choose backup based on current load/latency
2. **Model health scoring**: Track rolling success rates, prefer healthy backups
3. **Geographic failover**: Route to different regions if available
4. **Predictive quarantine**: Quarantine models showing degraded performance before complete failure
5. **Cost-aware routing**: Factor in token costs when selecting backups
