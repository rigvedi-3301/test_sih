import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_scheduler
)
from sklearn.metrics import accuracy_score
import wandb
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.login()

config = {
    "cysecbert_model": "markusbayer/CySecBERT",
    "electra_model": "google/electra-base-discriminator",
    "max_length": 128,
    "batch_size": 32,  
    "learning_rate": 3e-5,
    "epochs": 5,
    "train_split": 0.9,  
    "warmup_ratio": 0.085,  
    "scheduler_type": "linear",
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss",
    "weight_decay": 0.075,  
    "dropout_rate": 0.55,  
    "max_samples": 250000,
}

wandb.init(project="cysecbert_electra_fusion", name="benign_only_250k_gpu", config=config)

print("🔍 Checking GPU availability...")

if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU NOT FOUND! Training requires GPU. Please check your CUDA installation.")
    
device = torch.device("cuda")
print(f"✅ GPU found: {torch.cuda.get_device_name()}")
print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

torch.cuda.empty_cache()

print(f"Using device: {device}")

df = pd.read_csv("dataset_1.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['result'].value_counts()}")

if set(df['result'].unique()) != {0}:
    print("⚠️ Warning: Dataset contains non-benign labels. Filtering to only benign (0)...")
    df = df[df['result'] == 0].reset_index(drop=True)

print(f"After filtering - Dataset shape: {df.shape}")

if len(df) > config["max_samples"]:
    print(f"📊 Limiting dataset to {config['max_samples']} samples...")
    df = df.sample(n=config["max_samples"], random_state=42).reset_index(drop=True)
else:
    print(f"📊 Using all {len(df)} available samples (less than {config['max_samples']})")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

texts = df["url"].astype(str)
labels = torch.tensor(df["result"].values, dtype=torch.long) 

print(f"Training on {len(texts)} benign samples")

cysec_tokenizer = AutoTokenizer.from_pretrained(config["cysecbert_model"])
electra_tokenizer = AutoTokenizer.from_pretrained(config["electra_model"])

print("🔄 Tokenizing texts...")

def tokenize_in_chunks(tokenizer, texts, max_length, chunk_size=10000):
    """Tokenize in chunks to avoid memory issues"""
    all_input_ids = []
    all_attention_mask = []
    
    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i:i + chunk_size]
        encodings = tokenizer(
            list(chunk_texts), 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        all_input_ids.append(encodings["input_ids"])
        all_attention_mask.append(encodings["attention_mask"])
        
    return torch.cat(all_input_ids), torch.cat(all_attention_mask)

cysec_ids, cysec_mask = tokenize_in_chunks(cysec_tokenizer, texts, config["max_length"])
electra_ids, electra_mask = tokenize_in_chunks(electra_tokenizer, texts, config["max_length"])

print("💾 Keeping data on CPU, will move to GPU during training...")
dataset = TensorDataset(
    cysec_ids, 
    cysec_mask,
    electra_ids, 
    electra_mask,
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
    drop_last=True,
    pin_memory=True,  
    num_workers=2,    
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=config["batch_size"],
    pin_memory=True,
    num_workers=2,
)

class CySecElectraFusion(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name, dropout_rate=0.3):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        hidden_size = self.cysec.config.hidden_size + self.electra.config.hidden_size
        
       self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(256, 128),       
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(128, 2)  
        )
        
        for param in list(self.cysec.encoder.layer[:6].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:6].parameters()):
            param.requires_grad = False

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cysec_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:,0,:]
        electra_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:,0,:]
        combined = torch.cat((cysec_out, electra_out), dim=1)
        return self.classifier(combined)

model = CySecElectraFusion(
    config["cysecbert_model"], 
    config["electra_model"],
    dropout_rate=config["dropout_rate"]
).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)
loss_fn = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler('cuda')

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

print("🚀 Starting benign-only training on 250k samples (GPU ONLY)...")

for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, batch in enumerate(train_loader):
        cysec_ids, cysec_mask, electra_ids, electra_mask, lbls = [b.to(device, non_blocking=True) for b in batch]
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
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

        if (batch_idx + 1) % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

    avg_loss = total_loss / len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0
    all_preds, all_lbls = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            cysec_ids, cysec_mask, electra_ids, electra_mask, lbls = [b.to(device, non_blocking=True) for b in batch]
            
            with torch.amp.autocast('cuda'):
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
        torch.save(model.state_dict(), "best_model_benign_250k.pth")
        print("  💾 Saved best model")
    else:
        patience_counter += 1
        print(f"  ⚠️ Early stopping counter: {patience_counter}/{patience}")
        
    if patience_counter >= patience:
        print("  🛑 Early stopping triggered!")
        break

model.load_state_dict(torch.load("best_model_benign_250k.pth", map_location=device))

os.makedirs("cysec_electra_fusion_model_benign_250k", exist_ok=True)
torch.save(model.state_dict(), "cysec_electra_fusion_model_benign_250k/pytorch_model.bin")

cysec_tokenizer.save_pretrained("cysec_electra_fusion_model_benign_250k/cysec_tokenizer")
electra_tokenizer.save_pretrained("cysec_electra_fusion_model_benign_250k/electra_tokenizer")

import json
with open("cysec_electra_fusion_model_benign_250k/training_config.json", "w") as f:
    json.dump(config, f, indent=2)

wandb.save("cysec_electra_fusion_model_benign_250k/*")

print("✅ Benign-only training on 250k samples complete!")
print(f"📊 Final metrics - Best Val Loss: {best_val_loss:.4f}, Final Val Acc: {val_acc:.4f}")
print(f"💾 Model saved as: cysec_electra_fusion_model_benign_250k")

torch.cuda.empty_cache()
print("🧹 GPU memory cleaned up")
