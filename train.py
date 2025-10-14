import os, json, warnings
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoTokenizer, AutoModel, get_scheduler
import wandb
from torch.cuda.amp import autocast, GradScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")

wandb.login()

config = {
    "cysecbert_model": "markusbayer/CySecBERT",
    "electra_model": "google/electra-base-discriminator",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-6,
    "epochs": 2,
    "train_split": 0.9,
    "warmup_ratio": 0.06,
    "scheduler_type": "cosine",
    "optimizer": "AdamW",
    "weight_decay": 0.05,
    "dropout_rate": 0.4,
    "max_samples": 250000,
}

wandb.init(
    project="cysec_electra_oneclass",
    name="fusion_autoencoder_benign_only_fixed",
    config=config
)

if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU NOT FOUND! Please check CUDA.")
device = torch.device("cuda")
print(f"✅ GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
torch.cuda.empty_cache()

print("📁 Loading dataset...")
df = pd.read_csv("dataset_1.csv")
print(f"Loaded dataset shape: {df.shape}")

if "result" not in df.columns or "url" not in df.columns:
    raise ValueError("❌ dataset_1.csv must contain columns: 'url' and 'result'")

df = df[df["result"] == 0].reset_index(drop=True)
print(f"Filtered benign samples: {len(df)}")

if len(df) > config["max_samples"]:
    df = df.sample(n=config["max_samples"], random_state=42).reset_index(drop=True)
    print(f"Limited to {len(df)} samples")

texts = df["url"].astype(str)

print("🔤 Loading tokenizers...")
cysec_tokenizer = AutoTokenizer.from_pretrained(config["cysecbert_model"])
electra_tokenizer = AutoTokenizer.from_pretrained(config["electra_model"])

def tokenize_in_chunks(tokenizer, texts, max_length, chunk_size=5000):
    all_ids, all_masks = [], []
    total_chunks = (len(texts) - 1) // chunk_size + 1
    for i in range(0, len(texts), chunk_size):
        chunk_num = i // chunk_size + 1
        print(f"  Tokenizing chunk {chunk_num}/{total_chunks}...")
        chunk = texts[i:i + chunk_size]
        enc = tokenizer(list(chunk), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
    print("  ✅ Tokenization complete")
    return torch.cat(all_ids), torch.cat(all_masks)

print("🔄 Tokenizing with CySecBERT...")
cysec_ids, cysec_mask = tokenize_in_chunks(cysec_tokenizer, texts, config["max_length"])

print("🔄 Tokenizing with ELECTRA...")
electra_ids, electra_mask = tokenize_in_chunks(electra_tokenizer, texts, config["max_length"])

print("📊 Creating dataset...")
dataset = TensorDataset(cysec_ids, cysec_mask, electra_ids, electra_mask)
train_size = int(config["train_split"] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=0, pin_memory=False)

class FusionEncoder(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name):
        super().__init__()
        print("🔄 Loading CySecBERT model...")
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        print("🔄 Loading ELECTRA model...")
        self.electra = AutoModel.from_pretrained(electra_model_name)
        self.out_dim = self.cysec.config.hidden_size + self.electra.config.hidden_size
        for param in list(self.cysec.encoder.layer[:8].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:8].parameters()):
            param.requires_grad = False
        print(f"✅ Fusion encoder created with output dim: {self.out_dim}")

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cy_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:, 0, :]
        el_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:, 0, :]
        return torch.cat((cy_out, el_out), dim=1)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Tanh()
        )
        print(f"✅ Autoencoder created with input dim: {input_dim}")

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

print("🧠 Initializing models...")
fusion_encoder = FusionEncoder(config["cysecbert_model"], config["electra_model"]).to(device)
autoencoder = AutoEncoder(fusion_encoder.out_dim, dropout_rate=config["dropout_rate"]).to(device)

params = list(fusion_encoder.parameters()) + list(autoencoder.parameters())
optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = get_scheduler(config["scheduler_type"], optimizer, num_warmup_steps=int(config["warmup_ratio"] * len(train_loader) * config["epochs"]), num_training_steps=len(train_loader) * config["epochs"])
criterion = nn.MSELoss()
scaler = GradScaler()

print(f"🚀 Starting training with {len(train_loader)} batches per epoch...")

for epoch in range(config["epochs"]):
    fusion_encoder.train()
    autoencoder.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        cysec_ids, cysec_mask, electra_ids, electra_mask = [b.to(device, non_blocking=False) for b in batch]
        optimizer.zero_grad(set_to_none=True)
        
        fused = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
        reconstructed, _ = autoencoder(fused)
        loss = criterion(reconstructed, fused.detach())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    val_loss = 0.0
    fusion_encoder.eval()
    autoencoder.eval()
    with torch.no_grad():
        for batch in val_loader:
            cysec_ids, cysec_mask, electra_ids, electra_mask = [b.to(device, non_blocking=False) for b in batch]
            fused = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
            reconstructed, _ = autoencoder(fused)
            val_loss += criterion(reconstructed, fused).item()
    
    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f"📊 Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
    wandb.log({"epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val, "lr": scheduler.get_last_lr()[0]})

print("💾 Saving model...")
torch.save({
    "fusion_encoder": fusion_encoder.state_dict(),
    "autoencoder": autoencoder.state_dict()
}, "cysec_electra_oneclass.pth")

os.makedirs("cysec_electra_oneclass_model", exist_ok=True)
cysec_tokenizer.save_pretrained("cysec_electra_oneclass_model/cysec_tokenizer")
electra_tokenizer.save_pretrained("cysec_electra_oneclass_model/electra_tokenizer")

with open("cysec_electra_oneclass_model/training_config.json", "w") as f:
    json.dump(config, f, indent=2)

wandb.save("cysec_electra_oneclass_model/*")
print("✅ One-class benign-only training complete!")
