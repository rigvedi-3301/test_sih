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
import numpy as np

wandb.login()

config = {
    "cysecbert_model": "markusbayer/CySecBERT",
    "electra_model": "google/electra-base-discriminator",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 5,
    "train_split": 0.8,  
    "warmup_ratio": 0.85,
    "scheduler_type": "linear",
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss",
    "weight_decay": 0.01,  
    "dropout_rate": 0.5,  
}

wandb.init(project="cysecbert_electra_fusion", name="benign_only_training", config=config)

df = pd.read_csv("dataset_1.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['result'].value_counts()}")

if set(df['result'].unique()) != {0}:
    print("⚠️ Warning: Dataset contains non-benign labels. Filtering to only benign (0)...")
    df = df[df['result'] == 0].reset_index(drop=True)

print(f"After filtering - Dataset shape: {df.shape}")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

texts = df["url"].astype(str)
labels = torch.tensor(df["result"].values, dtype=torch.long) 

print(f"Training on {len(texts)} benign samples")

cysec_tokenizer = AutoTokenizer.from_pretrained(config["cysecbert_model"])
electra_tokenizer = AutoTokenizer.from_pretrained(config["electra_model"])

cysec_enc = cysec_tokenizer(
    list(texts), 
    padding=True, 
    truncation=True, 
    max_length=config["max_length"], 
    return_tensors="pt"
)
electra_enc = electra_tokenizer(
    list(texts), 
    padding=True, 
    truncation=True, 
    max_length=config["max_length"], 
    return_tensors="pt"
)

dataset = TensorDataset(
    cysec_enc["input_ids"], 
    cysec_enc["attention_mask"],
    electra_enc["input_ids"], 
    electra_enc["attention_mask"],
    labels
)

train_size = int(config["train_split"] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=config["batch_size"], 
    shuffle=True,
    drop_last=True  
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=config["batch_size"]
)

class CySecElectraFusion(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name, dropout_rate=0.5):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        hidden_size = self.cysec.config.hidden_size + self.electra.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  
        )
        
        for param in list(self.cysec.encoder.layer[:4].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:4].parameters()):
            param.requires_grad = False

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cysec_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:,0,:]
        electra_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:,0,:]
        combined = torch.cat((cysec_out, electra_out), dim=1)
        return self.classifier(combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CySecElectraFusion(
    config["cysecbert_model"], 
    config["electra_model"],
    dropout_rate=config["dropout_rate"]
).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

optimizer = AdamW(
    model.parameters(), 
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()

total_steps = len(train_loader) * config["epochs"]
warmup_steps = int(config["warmup_ratio"] * total_steps)
scheduler = get_scheduler(
    config["scheduler_type"], 
    optimizer=optimizer,
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

best_val_loss = float('inf')
patience = 2
patience_counter = 0

print("🚀 Starting benign-only training...")

for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        cysec_ids, cysec_mask, electra_ids, electra_mask, lbls = [b.to(device) for b in batch]
        optimizer.zero_grad()

        with autocast():
            logits = model(cysec_ids, cysec_mask, electra_ids, electra_mask)
            loss = loss_fn(logits, lbls)

        scaler.scale(loss).backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == lbls).sum().item()
        train_total += lbls.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0
    all_preds, all_lbls = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            cysec_ids, cysec_mask, electra_ids, electra_mask, lbls = [b.to(device) for b in batch]
            
            logits = model(cysec_ids, cysec_mask, electra_ids, electra_mask)
            loss = loss_fn(logits, lbls)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(all_lbls, all_preds)
    
    print(f"Epoch {epoch+1}/{config['epochs']}")
    print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    wandb.log({
        "epoch": epoch+1, 
        "train_loss": avg_loss, 
        "train_accuracy": train_acc,
        "val_loss": avg_val_loss, 
        "val_accuracy": val_acc,
        "learning_rate": scheduler.get_last_lr()[0]
    })
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model_benign.pth")
        print("  💾 Saved best model")
    else:
        patience_counter += 1
        print(f"  ⚠️ Early stopping counter: {patience_counter}/{patience}")
        
    if patience_counter >= patience:
        print("  🛑 Early stopping triggered!")
        break

model.load_state_dict(torch.load("best_model_benign.pth"))

model.save_pretrained("cysec_electra_fusion_model_benign")
wandb.save("cysec_electra_fusion_model_benign/*")

print("✅ Benign-only training complete! Model saved as 'cysec_electra_fusion_model_benign'")
print(f"📊 Final metrics - Best Val Loss: {best_val_loss:.4f}, Final Val Acc: {val_acc:.4f}")
