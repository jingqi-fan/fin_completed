import requests

# 你的 API KEY 和 URL
API_URL = "https://api.suanli.cn/v1/chat/completions"
API_KEY = "sk-W0rpStc95T7JVYVwDYc29IyirjtpPPby6SozFMQr17m8KWeo"

# 你要测试的 prompt
price_str = "100, 99, 98"
prompt = f"""You are an experienced investor holding multiple stocks. You usually don't check the push notifications for small price fluctuations, but you are more inclined to check them when there is a significant change.

Here are some examples:

Example 1:
The previous stock price was x USD.
Question: The system pushed a news that the current stock price changed to x+0.0005x USD. Would you like to click and view it?
Answer: No

Example 2:
The previous stock price was x USD.
Question: The system pushed a news that the current stock price changed to x+0.5x USD. Would you like to click and view it?
Answer: Yes

Example 3:
The previous stock price was x USD.
Question: The system pushed a news that the current stock price changed to x-0.0005x USD. Would you like to click and view it?
Answer: No

Example 4:
The previous stock price was x USD.
Question: The system pushed a news that the current stock price changed to x-0.5x USD. Would you like to click and view it?
Answer: Yes

Now please judge the new case:

The previous stock price was 100 USD.
Question: The system pushed a news that the current stock price changed to 99 USD. Would you like to click and view it? Please only answer "Yes" or "No".

Answer:
"""

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "free:QwQ-32B",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.3,
    "max_tokens": 50
}

response = requests.post(API_URL, headers=headers, json=data)

# 检查状态
if response.status_code == 200:
    res_json = response.json()
    content = res_json["choices"][0]["message"]["content"]
    print("LLM Response:", content)
else:
    print("Error:", response.status_code, response.text)
