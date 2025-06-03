import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW
from tqdm import tqdm
print(f'1')
from src.prepare_data import load_data
from src.model import build_model
print(f'2')

# ✅ GPU 设置（默认使用第0块卡）
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'cuda available: {torch.cuda.is_available()}')

# ✅ 超参数
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "output")

# ✅ 数据加载
train_dataset, val_dataset, tokenizer = load_data()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ✅ 构建模型、优化器、AMP
tokenizer.save_pretrained(SAVE_PATH)
model = build_model(device=device)
optimizer = AdamW(model.parameters(), lr=LR)
scaler = GradScaler()

# ✅ 训练循环（不含验证）
def train():
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with autocast():
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                loss = torch.nn.functional.cross_entropy(logits, batch['labels'])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f}")

    # ✅ 训练完成后保存模型
    torch.save(model.state_dict(),  "/home/zlwang/profile/output/pytorch_model.bin")


if __name__ == "__main__":
    train()