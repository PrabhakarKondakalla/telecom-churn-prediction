import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen:3.5",
        "prompt": "Explain AI simply",
        "stream": False   # 👈 IMPORTANT
    }
)

data = response.json()

print(data.get('response', 'No response key found'))