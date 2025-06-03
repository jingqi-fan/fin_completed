import requests
import json

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
API_KEY = "EMPTY"  # vllm默认不用API密钥

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# DeepSeek-V2 prompt 格式：用chat格式即可
data = {
    "model": "deepseek-v2-lite",
    "messages": [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "用简单的语言解释一下强化学习。"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": False
}

response = requests.post(API_URL, headers=headers, data=json.dumps(data))
print(response.json())
