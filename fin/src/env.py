import numpy as np

class StockEnvWithLLM:
    def __init__(self, price_seq, model, tokenizer, seed=None, window_size=1):
        self.price_seq = price_seq
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size  # 看多少天历史
        self.max_steps = len(price_seq) - 1
        self.seed(seed)

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.current_step = self.window_size  # 先跳过前几天，避免前几天历史不足
        return self._get_state(), {}

    def _get_state(self):
        # 这里返回最近window_size天的股价作为状态，更丰富
        state = self.price_seq[self.current_step - self.window_size:self.current_step]
        return np.array(state, dtype=np.float32)

    def _build_prompt(self, action):
        # 取前一时刻和当前时刻价格
        current_price_before = self.price_seq[self.current_step - 1]
        current_price_now = self.price_seq[self.current_step]

        prompt = f"""你是一个经验丰富的成熟投资者。对于股价的小幅波动通常不会查看推送，但当股价出现明显变化时会更倾向于点击查看。
        
        系统推送了一条新闻：该股票的股价从 {current_price_before:.2f} 元变为 {current_price_now:.2f} 元。那么你会点击查看吗？

        回答：
        """
        return prompt

    def _query_llm(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response

    def step(self, action):
        prompt = self._build_prompt(action)
        response = self._query_llm(prompt)
        # print(f'response: {response}')
        re = self.parse_response(response)
        # print(f're: {re}')
        if "不会" == re:
            if action == 1:
                reward = -1
            else:
                reward = 1
        elif "会" == re:
            if action == 1:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0

        self.current_step += 1
        done = self.current_step >= self.max_steps  # 重点：max_steps
        next_state = self._get_state()
        return next_state, reward, done, False, {}

    def parse_response(self, response):
        idx = response.find("点击查看吗？")
        if idx != -1:
            answer_part = response[idx + len("点击查看吗？"):]  # 裁剪出其后文本
        else:
            return "未知"
        # 去掉前后空白
        answer_part = answer_part.strip()
        # 规则匹配
        if "不会" in answer_part:
            return "不会"
        elif "会" in answer_part:
            return "会"
        else:
            return "未知"

    @property
    def observation_space(self):
        class Space:
            shape = (self.window_size,)
        return Space()

    @property
    def action_space(self):
        class Space:
            n = 2
        return Space()
