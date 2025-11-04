#!/usr/bin/env python3
"""Example usage of the MoE Ollama Endpoint."""
import asyncio
import httpx


async def main():
    """Demonstrate various API features."""
    base_url = "http://localhost:8000/v1"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("=" * 60)
        print("MoE Ollama Endpoint - Example Usage")
        print("=" * 60)
        
        # 1. Health check
        print("\n1. Health Check")
        response = await client.get(f"{base_url}/health")
        print(f"   Status: {response.json()}")
        
        # 2. List models
        print("\n2. List Available Models")
        response = await client.get(f"{base_url}/models")
        models = response.json()
        print(f"   Found {len(models['data'])} models:")
        for model in models['data'][:5]:
            print(f"   - {model['id']}")
        
        # 3. Simple chat completion
        print("\n3. Simple Chat Completion (Auto-routing)")
        response = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "What is Python?"}
                ],
                "temperature": 0.7,
            }
        )
        result = response.json()
        print(f"   Model used: {result['model']}")
        print(f"   Response: {result['choices'][0]['message']['content'][:100]}...")
        
        # 4. Code-specific query (should route to code model)
        print("\n4. Code Query (Should route to code model)")
        response = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
                ],
            }
        )
        result = response.json()
        print(f"   Model used: {result['model']}")
        print(f"   Response: {result['choices'][0]['message']['content'][:150]}...")
        
        # 5. Generate embeddings
        print("\n5. Generate Embeddings")
        response = await client.post(
            f"{base_url}/embeddings",
            json={
                "model": "nomic-embed-text",
                "input": "This is a test sentence for embeddings"
            }
        )
        result = response.json()
        embedding_dim = len(result['data'][0]['embedding'])
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   First 5 values: {result['data'][0]['embedding'][:5]}")
        
        # 6. RAG - Ingest documents
        print("\n6. RAG - Ingest Documents")
        response = await client.post(
            f"{base_url}/rag/ingest",
            json={
                "documents": [
                    {
                        "content": "The MoE Ollama Endpoint is a production-grade API that provides intelligent routing to multiple AI models.",
                        "metadata": {"source": "documentation", "version": "0.1.0"},
                        "collection": "docs"
                    },
                    {
                        "content": "FastAPI is a modern, fast web framework for building APIs with Python.",
                        "metadata": {"source": "tech", "topic": "web"},
                        "collection": "docs"
                    }
                ]
            }
        )
        result = response.json()
        print(f"   Ingested: {result['documents_ingested']} documents")
        print(f"   Message: {result['message']}")
        
        # 7. RAG - Search documents
        print("\n7. RAG - Search Documents")
        response = await client.get(
            f"{base_url}/rag/search",
            params={"query": "What is the MoE endpoint?", "top_k": 2}
        )
        results = response.json()
        print(f"   Found {len(results['results'])} similar documents:")
        for i, doc in enumerate(results['results'], 1):
            print(f"   {i}. Similarity: {doc['similarity_score']:.3f}")
            print(f"      Content: {doc['content'][:80]}...")
        
        # 8. RAG-enabled chat
        print("\n8. RAG-Enabled Chat Completion")
        response = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "What is the MoE Ollama Endpoint?"}
                ],
                "use_rag": True,
                "rag_collections": ["docs"]
            }
        )
        result = response.json()
        print(f"   Model used: {result['model']}")
        print(f"   Response: {result['choices'][0]['message']['content'][:200]}...")
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
