import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer
from src.prepare_data import load_data
from src.model import BertClassifier

# ✅ 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ✅ 路径配置
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "output")
MODEL_NAME = "/home/zlwang/profile/bert"  # 使用本地预训练模型

# ✅ 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# ✅ 加载模型并加载权重
model = BertClassifier(pretrained_model_name=MODEL_NAME)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location=device))
model.to(device)
model.eval()

# ✅ 加载验证集
_, val_dataset, _ = load_data()
val_loader = DataLoader(val_dataset, batch_size=32)

# ✅ 验证逻辑
def evaluate():
    preds = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            pred_labels = torch.argmax(logits, dim=1)
            preds.extend(pred_labels.cpu().tolist())
            targets.extend(batch['labels'].cpu().tolist())

    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, digits=4, target_names=["初级", "中级", "高级"])
    print(f"\n✅ 验证集准确率: {acc * 100:.2f}%")
    print("✅ 分类报告:")
    print(report)

if __name__ == "__main__":
    evaluate()
