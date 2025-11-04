#!/usr/bin/env python3
"""Test the OpenAI compatibility."""
import openai

# Configure to use local endpoint
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key for Ollama is configured server-side
)

print("Testing OpenAI Compatibility\n")

# Test 1: List models
print("1. Listing models...")
models = client.models.list()
print(f"   Available models: {len(models.data)}")
for model in models.data[:3]:
    print(f"   - {model.id}")

# Test 2: Simple completion
print("\n2. Simple chat completion...")
response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
)
print(f"   Response: {response.choices[0].message.content}")

# Test 3: Multi-turn conversation
print("\n3. Multi-turn conversation...")
messages = [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "What is 15 + 27?"},
]
response = client.chat.completions.create(
    model="auto",
    messages=messages
)
assistant_reply = response.choices[0].message.content
print(f"   Assistant: {assistant_reply}")

messages.append({"role": "assistant", "content": assistant_reply})
messages.append({"role": "user", "content": "Now multiply that by 2"})

response = client.chat.completions.create(
    model="auto",
    messages=messages
)
print(f"   Assistant: {response.choices[0].message.content}")

# Test 4: Embeddings
print("\n4. Creating embeddings...")
response = client.embeddings.create(
    model="nomic-embed-text",
    input="Hello, world!"
)
print(f"   Embedding dimension: {len(response.data[0].embedding)}")

print("\n✅ All OpenAI compatibility tests passed!")
