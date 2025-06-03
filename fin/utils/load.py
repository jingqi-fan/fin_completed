from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer():
    model_path = "/home/zlwang/fin_new/models/qwen/"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

    return model, tokenizer

def load_model_and_tokenizer_tinyllama():
    model_path = "/home/zlwang/fin_new/models/tinyllama/"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    return model, tokenizer

def parse_response_here(response):
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

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()

    prompt = f"""你是一个经验丰富的成熟投资者。对于股价的小幅波动通常不会查看推送，但当股价出现明显变化时会更倾向于点击查看。
    
    问题：系统推送了当前该股票股价从 101.7863272929448 元变为 96.94521370762007 元的新闻，那么你会点击查看吗？

    回答：
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print(response)
    print(f'now put response into clip')

    re = parse_response_here(response)
    print(f're = {re}')

    # model, tokenizer = load_model_and_tokenizer_tinyllama()
    # prompt = f"""You are an experienced investor who holds multiple stocks. You usually do not click on push notifications for small stock price changes, but you are more likely to click when there is a significant price change.
    #
    # Here are some examples:
    #
    # Example 1:
    # Question: The system is pushing a notification that the stock price has changed from x to x + 0.0005x USD. Would you like to click to view it?
    # Answer: No
    #
    # Example 2:
    # Question: The system is pushing a notification that the stock price has changed from x to x + 0.5x USD. Would you like to click to view it?
    # Answer: Yes
    #
    # Example 3:
    # Question: The system is pushing a notification that the stock price has changed from x to x - 0.0005x USD. Would you like to click to view it?
    # Answer: No
    #
    # Example 4:
    # Question: The system is pushing a notification that the stock price has changed from x to x - 0.5x USD. Would you like to click to view it?
    # Answer: Yes
    #
    # Example 5:
    # Question: The system is pushing a notification that the stock price has changed from 100 to 99 USD. Would you like to click to view it?
    # Answer:
    #
    # """
    #
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    # print("1"+response)

