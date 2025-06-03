import requests

class LLM_API_Client:
    def __init__(self, api_key):
        self.api_url = "https://api.suanli.cn/v1/chat/completions"
        self.api_key = api_key
        self.model_name = "free:QwQ-32B"

    def query(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 50,
            "stop": None
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()

        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"].strip()

        return content
