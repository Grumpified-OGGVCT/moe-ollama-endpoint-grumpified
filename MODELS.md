# Ollama Cloud Models Verification

This document provides detailed verification and justification for the 9 Ollama Cloud models used in this MoE endpoint, verified as of November 4, 2025.

## Model Verification & Best-Use Alignment

| Cloud tag | Modality | Official Documentation | Why it fits the assigned use-case |
|-----------|----------|------------------------|-----------------------------------|
| **deepseek-v3.1:671b-cloud** | Text | Hybrid thinking/non-thinking MoE that can emit a `thinking` field (chain-of-thought traces) | Built-in "thinking" trace makes it ideal for **long-form reasoning, step-by-step explanations and workflows requiring audit trails** |
| **gpt-oss:20b-cloud** | Text | Supports `think` field (low/medium/high), smallest cloud model with fastest latency and lowest cost | Perfect as **ultra-low-latency fallback for generic questions** where high quality is not essential |
| **gpt-oss:120b-cloud** | Text | Same API as 20b but with 120B total params (~5B active) | Provides **deep, multi-turn reasoning** for enterprise-level or highly contextual queries |
| **kimi-k2:1t-cloud** | Text | State-of-the-art MoE with 32B activated parameters, 1T total. Hugging Face repo explicitly mentions tool-calling support | Large activated parameter count and **tool-calling/math-focused** capabilities make it ideal for **STEM/math-heavy or autonomous agentic workflows** |
| **qwen3-coder:480b-cloud** | Text | "The most agentic code model to date" with 256K context, reinforcement-learning-trained on code | **State-of-the-art code-generation/debugging expert** with massive context window |
| **glm-4.6:cloud** | Text | "Thinking + tool-use" model supporting both modes, advanced agentic reasoning and coding with 198K context | Strong **tool-calling** and **structured-output** support make it ideal as **aggregator** synthesizing multiple expert replies into JSON-compatible answers |
| **minimax-m2:cloud** | Text | "High-efficiency large language model built for coding and agentic workflows" | **Cost-effective alternative** when budget is limited but coding/agentic response is needed |
| **qwen3-vl:235b-cloud** | Vision + Text | "Most powerful vision-language model in Qwen family" with Visual Agent capabilities (GUI recognition, tool calling, multimodal reasoning) | Use when **image is supplied** but **not** needing extra "thinking" optimization - e.g., simple captioning, visual QA, UI element extraction |
| **qwen3-vl:235b-instruct-cloud** | Vision + Text (instruct) | Same visual abilities plus "Stronger Multimodal Reasoning (Thinking Version)" optimized for STEM and math reasoning | Use when request combines **image plus math/code problem** (e.g., "solve the diagram" or "generate code from this UI mock-up") |

## Key Capabilities

### Tool-Calling Support
Tool-calling is a **generic capability** of the Ollama API - any model supporting the `tools` field can request functions. Verified support:
- **Kimi-K2**: Explicit tool-calling guidance in Hugging Face docs
- **Vision models**: Described as "Visual Agent" with tool invocation capabilities
- **GLM-4.6**: "Thinking + tool-use" model with native support

### Thinking Mode
Models supporting chain-of-thought reasoning:
- `deepseek-v3.1:671b-cloud` - Hybrid thinking/non-thinking with explicit `thinking` field
- `gpt-oss:20b-cloud` and `gpt-oss:120b-cloud` - Support `think` field with levels (low/medium/high)
- `qwen3-vl:235b-instruct-cloud` - Thinking-optimized for visual reasoning

### Context Lengths
- `qwen3-coder:480b-cloud` - 256K tokens (262,144)
- `glm-4.6:cloud` - 198K tokens
- Standard models - Varies by implementation

### MoE Efficiency
All models use sparse activation:
- Typical activation: 10-37B parameters per inference
- Total parameters: 20B to 1T depending on model
- Cost-effective through sparse MoE architecture

## References

- [Ollama Cloud Models Documentation](https://ollama.com/blog/cloud-models)
- [Ollama Tool Calling Documentation](https://docs.ollama.com/capabilities/tool-calling)
- Individual model pages on Ollama.com
- Hugging Face model repositories
- [Skywork.ai Ollama Models Comparison](https://skywork.ai/blog/llm/ollama-models-list-2025-100-models-compared/)
