import os
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

MODEL_NAME = "/home/zlwang/profile/bert"  # 本地模型路径
MAX_LEN = 512  # 支持多轮对话拼接

class UserLevelDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item['text'], list):
            text = " [SEP] ".join(item['text'][-5:])  # 最近 5 条对话
        else:
            text = item['text']

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def load_data():
    train_path = os.path.join(os.path.dirname(__file__), "..", "data", "data_train.json")
    val_path = os.path.join(os.path.dirname(__file__), "..", "data", "data_validate.json")

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = UserLevelDataset(train_data, tokenizer)
    val_dataset = UserLevelDataset(val_data, tokenizer)

    return train_dataset, val_dataset, tokenizer
