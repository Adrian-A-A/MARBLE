import os
from litellm import completion

# This mimics how MARBLE uses LiteLLM under the hood
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

try:
    response = completion(
        model="openai/qwen2.5:0.5b", # LiteLLM needs the 'openai/' prefix to use the base_url
        messages=[{"content": "respond with 'Connection Successful'", "role": "user"}]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")