"""Tests for the MoE routing logic with Ollama Cloud models."""
import pytest
from app.services.router import moe_router


class TestMoERouting:
    """Test suite for MoE routing decisions."""

    @pytest.mark.asyncio
    async def test_vision_routing_simple(self):
        """Test routing to vision model for simple image tasks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                ]
            }
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "qwen3-vl:235b-cloud"
        assert use_rag is False

    @pytest.mark.asyncio
    async def test_vision_routing_with_reasoning(self):
        """Test routing to vision+thinking model for complex visual reasoning."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image in detail and explain the reasoning"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                ]
            }
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "qwen3-vl:235b-instruct-cloud"
        assert use_rag is False

    @pytest.mark.asyncio
    async def test_code_routing_advanced(self):
        """Test routing to advanced code model."""
        messages = [
            {"role": "user", "content": "Write a complex function to implement a binary search tree"}
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "qwen3-coder:480b-cloud"
        assert use_rag is False

    @pytest.mark.asyncio
    async def test_code_routing_simple(self):
        """Test routing to cost-effective code model for simple tasks."""
        messages = [
            {"role": "user", "content": "Write a simple script to print hello world"}
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "minimax-m2:cloud"
        assert use_rag is False

    @pytest.mark.asyncio
    async def test_math_tool_routing(self):
        """Test routing to math/tool model."""
        messages = [
            {"role": "user", "content": "Calculate the integral of x^2 from 0 to 10"}
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "kimi-k2:1t-cloud"
        assert use_rag is False

    @pytest.mark.asyncio
    async def test_reasoning_routing(self):
        """Test routing to complex reasoning model."""
        messages = [
            {"role": "user", "content": "Analyze the complex implications of quantum computing on cryptography"}
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "deepseek-v3.1:671b-cloud"
        assert use_rag is False

    @pytest.mark.asyncio
    async def test_enterprise_routing(self):
        """Test routing to enterprise model."""
        messages = [
            {"role": "user", "content": "Provide an enterprise production analysis of market trends"}
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "gpt-oss:120b-cloud"
        assert use_rag is False

    @pytest.mark.asyncio
    async def test_aggregation_routing(self):
        """Test routing to aggregator model."""
        messages = [
            {"role": "user", "content": "Summarize and combine these multiple reports"}
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "glm-4.6:cloud"
        assert use_rag is False

    @pytest.mark.asyncio
    async def test_rag_routing(self):
        """Test routing with RAG enabled."""
        messages = [
            {"role": "user", "content": "Search the knowledge base for information about AI"}
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "gpt-oss:20b-cloud"
        assert use_rag is True

    @pytest.mark.asyncio
    async def test_default_routing(self):
        """Test default fallback routing."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        model, use_rag = await moe_router.route_request(messages)
        assert model == "gpt-oss:20b-cloud"
        assert use_rag is False

    def test_backup_chain(self):
        """Test backup chain configuration."""
        # Test reasoning model backup
        backups = moe_router.get_backup_models("deepseek-v3.1:671b-cloud")
        assert backups == ["gpt-oss:120b-cloud", "gpt-oss:20b-cloud"]
        
        # Test code model backup
        backups = moe_router.get_backup_models("qwen3-coder:480b-cloud")
        assert backups == ["minimax-m2:cloud", "gpt-oss:20b-cloud"]
        
        # Test vision model backup
        backups = moe_router.get_backup_models("qwen3-vl:235b-cloud")
        assert backups == ["qwen3-vl:235b-instruct-cloud", "gpt-oss:20b-cloud"]
        
        # Test fallback model backup
        backups = moe_router.get_backup_models("gpt-oss:20b-cloud")
        assert backups == ["gpt-oss:120b-cloud"]

    def test_model_info(self):
        """Test model info retrieval."""
        info = moe_router.get_model_info()
        
        # Check all models are present
        assert info["reasoning"] == "deepseek-v3.1:671b-cloud"
        assert info["fallback"] == "gpt-oss:20b-cloud"
        assert info["enterprise"] == "gpt-oss:120b-cloud"
        assert info["math_tool"] == "kimi-k2:1t-cloud"
        assert info["code"] == "qwen3-coder:480b-cloud"
        assert info["aggregator"] == "glm-4.6:cloud"
        assert info["cost_code"] == "minimax-m2:cloud"
        assert info["vision"] == "qwen3-vl:235b-cloud"
        assert info["vision_thinking"] == "qwen3-vl:235b-instruct-cloud"
        assert info["default"] == "gpt-oss:20b-cloud"
        
        # Check backup strategy is present
        assert "backup_strategy" in info
        expected_backup_count = len(info["backup_strategy"])
        assert expected_backup_count == 9  # Verify we have 9 models configured
    
    def test_circuit_breaker_quarantine(self):
        """Test circuit breaker quarantine functionality."""
        model = "deepseek-v3.1:671b-cloud"
        
        # Initially not quarantined
        assert not moe_router.is_quarantined(model)
        
        # Record failures up to threshold
        for _ in range(3):
            moe_router.record_failure(model)
        
        # Should now be quarantined
        assert moe_router.is_quarantined(model)
        
        # Reset for other tests
        moe_router.record_success(model)
        assert not moe_router.is_quarantined(model)
    
    def test_get_available_model_with_quarantine(self):
        """Test getting available model when primary is quarantined."""
        primary = "qwen3-coder:480b-cloud"
        
        # Quarantine primary
        for _ in range(3):
            moe_router.record_failure(primary)
        
        # Should return backup
        available = moe_router.get_available_model(primary)
        assert available != primary
        assert available in ["minimax-m2:cloud", "gpt-oss:20b-cloud"]
        
        # Clean up
        moe_router.record_success(primary)
    
    def test_strip_images_for_fallback(self):
        """Test stripping images for text-only fallback."""
        messages_with_images = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}
                ]
            }
        ]
        
        cleaned = moe_router.strip_images_for_fallback(messages_with_images)
        
        # Should have text content only
        assert len(cleaned) == 1
        assert cleaned[0]["role"] == "user"
        assert isinstance(cleaned[0]["content"], str)
        assert "Describe this image" in cleaned[0]["content"]
    
    def test_record_success_resets_failures(self):
        """Test that recording success resets failure count."""
        model = "gpt-oss:120b-cloud"
        
        # Record some failures
        moe_router.record_failure(model)
        moe_router.record_failure(model)
        assert model in moe_router.failure_counts
        
        # Record success
        moe_router.record_success(model)
        assert model not in moe_router.failure_counts
    
    def test_get_all_expert_models(self):
        """Test getting all expert models."""
        all_models = moe_router.get_all_expert_models()
        
        # Should have all 9 models
        assert len(all_models) == 9
        assert "deepseek-v3.1:671b-cloud" in all_models
        assert "qwen3-coder:480b-cloud" in all_models
        assert "qwen3-vl:235b-cloud" in all_models
    
    @pytest.mark.asyncio
    async def test_select_experts_for_query(self):
        """Test expert selection for a query."""
        messages = [{"role": "user", "content": "Write a Python function"}]
        
        # Select 2 experts (default)
        experts = await moe_router.select_experts_for_query(messages, k=2)
        assert len(experts) == 2
        # Should get 2 valid models (routing may vary)
        assert all(expert in moe_router.get_all_expert_models() for expert in experts)
    
    @pytest.mark.asyncio
    async def test_select_experts_expansion(self):
        """Test expert selection with k expansion."""
        messages = [{"role": "user", "content": "Explain quantum computing"}]
        
        # Select 5 experts
        experts = await moe_router.select_experts_for_query(messages, k=5)
        assert len(experts) == 5
        assert len(set(experts)) == len(experts)  # No duplicates
    
    def test_calculate_coverage_score_high_quality(self):
        """Test coverage score calculation for high quality responses."""
        responses = [
            "This is a comprehensive answer with detailed explanation of the topic.",
            "Here's another thorough response providing additional context and examples.",
        ]
        
        score = moe_router.calculate_coverage_score(responses)
        assert score > 0.6  # Should be reasonably high
    
    def test_calculate_coverage_score_low_confidence(self):
        """Test coverage score with low confidence phrases."""
        responses = [
            "I'm not sure about this, maybe it could be related to X.",
            "Perhaps this is correct, but I'm uncertain.",
        ]
        
        score = moe_router.calculate_coverage_score(responses)
        assert score < 0.5  # Should be lower due to uncertainty
    
    def test_calculate_coverage_score_short_responses(self):
        """Test coverage score with very short responses."""
        responses = ["Yes.", "No.", "Maybe."]
        
        score = moe_router.calculate_coverage_score(responses)
        assert score < 0.7  # Should be penalized for brevity
    
    def test_aggregate_expert_responses(self):
        """Test aggregation of multiple expert responses."""
        expert_responses = {
            "deepseek-v3.1:671b-cloud": "Expert 1 says this is correct.",
            "gpt-oss:120b-cloud": "Expert 2 provides additional context.",
        }
        
        aggregated = moe_router.aggregate_expert_responses(expert_responses)
        
        # Should contain both responses
        assert "Expert 1" in aggregated or "this is correct" in aggregated
        assert len(aggregated) > 0
    
    def test_aggregate_single_response(self):
        """Test aggregation of single expert response."""
        expert_responses = {
            "qwen3-coder:480b-cloud": "Here's the code implementation.",
        }
        
        aggregated = moe_router.aggregate_expert_responses(expert_responses)
        assert aggregated == "Here's the code implementation."
