import os, json, warnings
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, get_scheduler
import wandb
from torch.cuda.amp import GradScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
wandb.login()

config = {
    "cysecbert_model": "markusbayer/CySecBERT",
    "electra_model": "google/electra-base-discriminator",
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 3e-7,
    "epochs": 5,
    "train_split": 0.9,
    "warmup_ratio": 0.2,
    "scheduler_type": "cosine",
    "optimizer": "AdamW",
    "weight_decay": 0.2,
    "dropout_rate": 0.5,
    "max_samples": 300000,
    "freeze_layers": 8,
    "gradient_clip": 0.5,
    "label_smoothing": 0.1,
    "val_max_samples": 50000
}

wandb.init(project="cysec_electra_oneclass", name="fusion_autoencoder_benign_only_FIXED_v4", config=config)

if not torch.cuda.is_available():
    raise RuntimeError("GPU NOT FOUND!")
device = torch.device("cuda")
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name()}")

df = pd.read_csv("dataset_1.csv")
if "result" not in df.columns or "url" not in df.columns:
    raise ValueError("dataset_1.csv must contain columns: 'url' and 'result'")

df_benign = df[df["result"] == 0].reset_index(drop=True)
df_malicious = df[df["result"] == 1].reset_index(drop=True)

if len(df_benign) > config["max_samples"]:
    df_benign_train = df_benign.sample(n=config["max_samples"], random_state=42).reset_index(drop=True)
else:
    df_benign_train = df_benign.copy()

df_benign_val = df_benign.drop(df_benign_train.index).reset_index(drop=True)
df_val = pd.concat([df_benign_val, df_malicious]).reset_index(drop=True)
if len(df_val) > config["val_max_samples"]:
    df_val = df_val.sample(n=config["val_max_samples"], random_state=42).reset_index(drop=True)

train_texts = df_benign_train["url"].astype(str)
val_texts = df_val["url"].astype(str)

print(f"Training on {len(train_texts)} benign samples")
print(f"Validation on {len(val_texts)} URLs")

cysec_tokenizer = AutoTokenizer.from_pretrained(config["cysecbert_model"])
electra_tokenizer = AutoTokenizer.from_pretrained(config["electra_model"])

def tokenize_in_chunks(tokenizer, texts, max_length, chunk_size=5000):
    all_ids, all_masks = [], []
    total_chunks = (len(texts) - 1) // chunk_size + 1
    for i in range(0, len(texts), chunk_size):
        chunk_num = i // chunk_size + 1
        chunk = texts[i:i + chunk_size]
        enc = tokenizer(list(chunk), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
        print(f"Tokenized chunk {chunk_num}/{total_chunks}")
    return torch.cat(all_ids), torch.cat(all_masks)

cysec_train_ids, cysec_train_mask = tokenize_in_chunks(cysec_tokenizer, train_texts, config["max_length"])
cysec_val_ids, cysec_val_mask = tokenize_in_chunks(cysec_tokenizer, val_texts, config["max_length"])
electra_train_ids, electra_train_mask = tokenize_in_chunks(electra_tokenizer, train_texts, config["max_length"])
electra_val_ids, electra_val_mask = tokenize_in_chunks(electra_tokenizer, val_texts, config["max_length"])

train_dataset = TensorDataset(cysec_train_ids, cysec_train_mask, electra_train_ids, electra_train_mask)
val_dataset = TensorDataset(cysec_val_ids, cysec_val_mask, electra_val_ids, electra_val_mask)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

class FusionEncoder(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name, freeze_layers=8):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        self.out_dim = self.cysec.config.hidden_size + self.electra.config.hidden_size
        for param in list(self.cysec.encoder.layer[:freeze_layers].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:freeze_layers].parameters()):
            param.requires_grad = False
    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cy_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:,0,:]
        el_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:,0,:]
        return torch.cat((cy_out, el_out), dim=1)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128,64), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate*0.5),
            nn.Linear(128,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate*0.5),
            nn.Linear(256,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate*0.5),
            nn.Linear(512,input_dim), nn.Tanh()
        )
    def forward(self,x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

fusion_encoder = FusionEncoder(config["cysecbert_model"], config["electra_model"], freeze_layers=config["freeze_layers"]).to(device)
autoencoder = AutoEncoder(fusion_encoder.out_dim, dropout_rate=config["dropout_rate"]).to(device)

params = list(fusion_encoder.parameters()) + list(autoencoder.parameters())
optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = get_scheduler(config["scheduler_type"], optimizer,
                          num_warmup_steps=int(config["warmup_ratio"]*len(train_loader)*config["epochs"]),
                          num_training_steps=len(train_loader)*config["epochs"])
criterion = nn.MSELoss()
scaler = GradScaler()

for epoch in range(config["epochs"]):
    fusion_encoder.train()
    autoencoder.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader, 1):
        cysec_ids, cysec_mask, electra_ids, electra_mask = [b.to(device) for b in batch]
        optimizer.zero_grad(set_to_none=True)
        fused = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
        fused_noisy = fused + torch.randn_like(fused)*config["label_smoothing"]*0.1 if config.get("label_smoothing",0)>0 else fused
        reconstructed, _ = autoencoder(fused_noisy)
        loss = criterion(reconstructed, fused)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config["gradient_clip"])
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
    fusion_encoder.eval()
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            cysec_ids, cysec_mask, electra_ids, electra_mask = [b.to(device) for b in batch]
            fused = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
            reconstructed, _ = autoencoder(fused)
            val_loss += criterion(reconstructed, fused).item()
    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
    wandb.log({"epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val, "lr": scheduler.get_last_lr()[0]})

torch.save({"fusion_encoder": fusion_encoder.state_dict(), "autoencoder": autoencoder.state_dict()}, "cysec_electra_oneclass_v4.pth")
os.makedirs("cysec_electra_oneclass_model_v4", exist_ok=True)
cysec_tokenizer.save_pretrained("cysec_electra_oneclass_model_v4/cysec_tokenizer")
electra_tokenizer.save_pretrained("cysec_electra_oneclass_model_v4/electra_tokenizer")
with open("cysec_electra_oneclass_model_v4/training_config.json","w") as f:
    json.dump(config,f,indent=2)
wandb.save("cysec_electra_oneclass_model_v4/*")
print("One-class benign-only training complete!")
