import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_scheduler,
    AdamW
)
from sklearn.metrics import accuracy_score
import wandb
from torch.cuda.amp import autocast, GradScaler

wandb.login()

config = {
    "cysecbert_model": "markusbayer/CySecBERT",
    "electra_model": "google/electra-base-discriminator",
    "max_length": 256,
    "batch_size": 32,
    "learning_rate": 1.5e-5,
    "epochs": 3,
    "train_split": 0.9,
    "warmup_ratio": 0.1,
    "scheduler_type": "cosine",
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss"
}

wandb.init(project="cysecbert_electra_fusion", config=config)

df = pd.read_csv("./minitrain_data/csic_cleaned.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

texts = df["URL"] + " " + df["content"].fillna("") + " " + df["Method"].fillna("") + " " + df["User-Agent"].fillna("")
labels = torch.tensor(df["classification"].values, dtype=torch.long)

cysec_tokenizer = AutoTokenizer.from_pretrained(config["cysecbert_model"])
electra_tokenizer = AutoTokenizer.from_pretrained(config["electra_model"])

cysec_enc = cysec_tokenizer(list(texts), padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
electra_enc = electra_tokenizer(list(texts), padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")

dataset = TensorDataset(
    cysec_enc["input_ids"], cysec_enc["attention_mask"],
    electra_enc["input_ids"], electra_enc["attention_mask"],
    labels
)

train_size = int(config["train_split"] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

class CySecElectraFusion(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        hidden_size = self.cysec.config.hidden_size + self.electra.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cysec_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:,0,:]
        electra_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:,0,:]
        combined = torch.cat((cysec_out, electra_out), dim=1)
        return 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CySecElectraFusion(config["cysecbert_model"], config["electra_model"]).to(device)
optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()

total_steps = len(train_loader) * config["epochs"]
warmup_steps = int(config["warmup_ratio"] * total_steps)
scheduler = get_scheduler(
    config["scheduler_type"], optimizer=optimizer,
    num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0

    for batch in train_loader:
        cysec_ids, cysec_mask, electra_ids, electra_mask, lbls = [b.to(device) for b in batch]
        optimizer.zero_grad()

        with autocast():
            logits = model(cysec_ids, cysec_mask, electra_ids, electra_mask)
            loss = loss_fn(logits, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for batch in val_loader:
            cysec_ids, cysec_mask, electra_ids, electra_mask, lbls = [b.to(device) for b in batch]
            logits = model(cysec_ids, cysec_mask, electra_ids, electra_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.cpu().numpy())

    acc = accuracy_score(all_lbls, all_preds)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Accuracy: {acc:.4f}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss, "val_accuracy": acc})

model.save_pretrained("cysec_electra_fusion_model")
wandb.save("cysec_electra_fusion_model/*")
print("✅ Training complete! Model saved as 'cysec_electra_fusion_model'.")
