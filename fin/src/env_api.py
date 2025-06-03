import numpy as np

class StockEnvWithLLM_api:
    def __init__(self, price_seq, api_client, seed=None, window_size=3):
        self.price_seq = price_seq
        # self.model = model
        # self.tokenizer = tokenizer
        self.api_client = api_client
        self.window_size = window_size  # 看多少天历史
        self.max_steps = len(price_seq) - 1  # 完全兼容 gym 里的 max_episode_steps
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
        history_prices = self.price_seq[self.current_step - self.window_size:self.current_step]
        price_str = ", ".join([f"{p:.2f}" for p in history_prices])
        # prompt = f"""你是一个非常关注股价变化的投资者。你对股价上涨尤为敏感，更倾向于在股价上涨时点击查看股票动态；而在股价下跌时，虽然也会关注，但点击的意愿相对较低。过去几天股价为：{price_str} 元。系统向你推送了一条关于该股票的最新动态，请问你会点击查看吗？请只回答：是 或 否。"""
        prompt = f"""You are an investor who closely follows stock price changes. You are particularly sensitive to price increases and are more likely to click on stock updates when prices go up. Although you also pay attention when prices decrease, your willingness to click is relatively lower. The stock prices over the past few days are: {price_str} USD.

        The system is pushing you the latest update about this stock. Would you like to click and view it? Please answer with only "Yes" or "No"."""
        return prompt

    # def _query_llm(self, prompt):
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
    #     output = self.model.generate(**inputs, max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id)
    #     response = self.tokenizer.decode(output[0], skip_special_tokens=True)
    #
    #     return response

    def _query_llm(self, prompt):
        response = self.api_client.query(prompt)
        return response

    def step(self, action):
        prompt = self._build_prompt(action)
        response = self._query_llm(prompt)
        # print(f'response: {response}')

        if "不会" in response:
            if action == 1:
                reward = -1
            else:
                reward = 1
        elif "会" in response:
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
