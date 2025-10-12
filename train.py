import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_scheduler
)
from sklearn.metrics import accuracy_score
import wandb
from tqdm import tqdm

# ============================================================
# 1️⃣ WandB setup
# ============================================================
wandb.login()

config = {
    "cyberbert_model": "cybersecurityai/cyberbert-base",
    "electra_model": "google/electra-base-discriminator",
    "max_length": 256,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 5,
    "train_val_split": 0.8,
    "warmup_ratio": 0.1,
    "scheduler_type": "cosine",
    "fusion": "feature_concat"
}

wandb.init(project="cyberbert_electra_fusion", config=config, name="csic_train_run")

# ============================================================
# 2️⃣ Dataset class
# ============================================================
class CSICDataset(Dataset):
    def __init__(self, dataframe, tokenizer1, tokenizer2, max_len):
        self.data = dataframe
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # combine all meaningful textual fields
        text = f"{row['Method']} {row['User-Agent']} {row['URL']} {row['content']}"
        label = int(row['classification'])

        enc1 = self.tokenizer1(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        enc2 = self.tokenizer2(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )

        item = {
            "input_ids_1": enc1["input_ids"].squeeze(0),
            "attention_mask_1": enc1["attention_mask"].squeeze(0),
            "input_ids_2": enc2["input_ids"].squeeze(0),
            "attention_mask_2": enc2["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }
        return item

# ============================================================
# 3️⃣ Load data
# ============================================================
df = pd.read_csv("csic_cleaned.csv").dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# load both tokenizers
tokenizer1 = AutoTokenizer.from_pretrained(config["cyberbert_model"])
tokenizer2 = AutoTokenizer.from_pretrained(config["electra_model"])

dataset = CSICDataset(df, tokenizer1, tokenizer2, config["max_length"])
train_size = int(config["train_val_split"] * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

# ============================================================
# 4️⃣ Model Definition (Fusion Model)
# ============================================================
class CyberElectraFusion(nn.Module):
    def __init__(self, model1_name, model2_name, num_labels=2):
        super().__init__()
        self.model1 = AutoModel.from_pretrained(model1_name)
        self.model2 = AutoModel.from_pretrained(model2_name)

        hidden1 = self.model1.config.hidden_size
        hidden2 = self.model2.config.hidden_size
        fusion_dim = hidden1 + hidden2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        out1 = self.model1(input_ids=input_ids_1, attention_mask=attention_mask_1)
        out2 = self.model2(input_ids=input_ids_2, attention_mask=attention_mask_2)

        cls1 = out1.last_hidden_state[:, 0, :]
        cls2 = out2.last_hidden_state[:, 0, :]
        fused = torch.cat((cls1, cls2), dim=1)

        logits = self.classifier(fused)
        return logits

# ============================================================
# 5️⃣ Training setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CyberElectraFusion(config["cyberbert_model"], config["electra_model"])
model.to(device)

optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
loss_fn = nn.CrossEntropyLoss()

# scheduler
total_steps = len(train_loader) * config["epochs"]
warmup_steps = int(config["warmup_ratio"] * total_steps)
scheduler = get_scheduler(
    name=config["scheduler_type"],
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ============================================================
# 6️⃣ Training loop
# ============================================================
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in progress:
        b = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(**b)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            b = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            logits = model(**b)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Acc={acc:.4f}")
    wandb.log({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "val_accuracy": acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# ============================================================
# 7️⃣ Save model
# ============================================================
torch.save(model.state_dict(), "cyber_electra_fusion.pt")
wandb.save("cyber_electra_fusion.pt")
print("✅ Training complete — model saved as cyber_electra_fusion.pt")
