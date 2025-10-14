import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json
import os

class FusionEncoder(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        self.out_dim = self.cysec.config.hidden_size + self.electra.config.hidden_size
        for param in list(self.cysec.encoder.layer[:8].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:8].parameters()):
            param.requires_grad = False

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

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

model_path = "cysec_electra_oneclass_model"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model path not found: {model_path}")

with open(f"{model_path}/training_config.json", "r") as f:
    config = json.load(f)

cysec_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/cysec_tokenizer")
electra_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/electra_tokenizer")

fusion_encoder = FusionEncoder(config["cysecbert_model"], config["electra_model"])
autoencoder = AutoEncoder(fusion_encoder.out_dim, dropout_rate=config["dropout_rate"])

weights_path = "cysec_electra_oneclass.pth"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"❌ Model weights not found at: {weights_path}")

state_dict = torch.load(weights_path, map_location="cpu")
fusion_encoder.load_state_dict(state_dict["fusion_encoder"])
autoencoder.load_state_dict(state_dict["autoencoder"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_encoder.to(device)
autoencoder.to(device)
fusion_encoder.eval()
autoencoder.eval()

print(f"✅ Autoencoder model loaded successfully on: {device}")

test_urls = [
    "https://www.example.com/",
    "https://shop.example.com/product/12345?ref=google&utm_source=email",
    "https://docs.example.org/user-guide/v2.1/user_guide.pdf",
    "https://accounts.example.com/login?continue=/dashboard",
    "http://192.0.2.45/downloads/update.exe",
    "http://203.0.113.77/installer/latest_installer.zip?payload=cmd.exe",
    "https://www.example.com/%2e%2e/%2e%2e/admin/config.php",
    "https://login.example.com/?user=admin&pass=%3Cscript%3Ealert(1)%3C%2Fscript%3E",
    "https://google.com",
    "https://free-download-malware.ru",
    "https://www.bankofamerica.com/login",
    "http://bit.ly/2FakeLink",
    "http://192.168.1.1/admin"
]

print("🔄 Tokenizing URLs...")

cysec_encodings = cysec_tokenizer(
    test_urls,
    padding=True,
    truncation=True,
    max_length=config["max_length"],
    return_tensors="pt"
)
electra_encodings = electra_tokenizer(
    test_urls,
    padding=True,
    truncation=True,
    max_length=config["max_length"],
    return_tensors="pt"
)

cysec_ids = cysec_encodings["input_ids"].to(device)
cysec_mask = cysec_encodings["attention_mask"].to(device)
electra_ids = electra_encodings["input_ids"].to(device)
electra_mask = electra_encodings["attention_mask"].to(device)

print("🔮 Running anomaly detection...")

criterion = nn.MSELoss()
predictions = []

with torch.no_grad():
    fused = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
    reconstructed, _ = autoencoder(fused)
    
    for i in range(len(test_urls)):
        error = criterion(reconstructed[i], fused[i]).item()
        
        if error < 0.05:
            predictions.append("benign")
        elif error < 0.08:
            predictions.append("suspicious") 
        else:
            predictions.append("malicious")

print("\n" + "="*80)
print("🔒 URL Classification Results")
print("="*80)

benign_count = 0
suspicious_count = 0
malicious_count = 0

for url, pred in zip(test_urls, predictions):
    print(f"URL: {url}")
    
    if pred == "benign":
        print("✅ BENIGN")
        benign_count += 1
    elif pred == "suspicious":
        print("⚠️  SUSPICIOUS") 
        suspicious_count += 1
    else:
        print("🚨 MALICIOUS")
        malicious_count += 1
    print("-" * 80)

print(f"\n📊 Summary:")
print(f"Benign URLs: {benign_count}")
print(f"Suspicious URLs: {suspicious_count}") 
print(f"Malicious URLs: {malicious_count}")
print(f"Total URLs analyzed: {len(test_urls)}")
