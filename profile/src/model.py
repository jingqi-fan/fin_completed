import torch
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str = "/home/zlwang/profile/bert", num_labels: int = 3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

def build_model(model_name="/home/zlwang/profile/bert", num_labels=3, device="cuda"):
    model = BertClassifier(pretrained_model_name=model_name, num_labels=num_labels)
    return model.to(device)