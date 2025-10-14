import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json
import os
import numpy as np

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

print(f"✅ Model loaded successfully on: {device}")
print(f"📊 Model trained on: {config['max_samples']} benign samples")

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

print("🔮 Analyzing URL similarity to benign training patterns...")

with torch.no_grad():
    # Get embeddings for all URLs
    embeddings = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
    
    # Try to reconstruct them using the autoencoder
    reconstructed, _ = autoencoder(embeddings)
    
    # Calculate reconstruction error (how well autoencoder can recreate the embedding)
    reconstruction_errors = torch.mean((embeddings - reconstructed) ** 2, dim=1).cpu().numpy()

print("\n" + "="*80)
print("🔍 URL Similarity to Benign Training Patterns")
print("="*80)

# Convert reconstruction error to "benign similarity" score
# Lower error = more similar to training data
max_error = max(reconstruction_errors)
benign_similarities = [1.0 - (error / max_error) for error in reconstruction_errors]

benign_count = 0
malicious_count = 0

for url, error, similarity in zip(test_urls, reconstruction_errors, benign_similarities):
    print(f"URL: {url}")
    print(f"Reconstruction Error: {error:.6f}")
    print(f"Similarity to Benign: {similarity:.2%}")
    
    if similarity > 0.85:
        print("🟢 HIGHLY BENIGN")
        print("   ✓ Very similar to training patterns")
        benign_count += 1
    elif similarity > 0.70:
        print("🟡 LIKELY BENIGN") 
        print("   ✓ Similar to training patterns")
        benign_count += 1
    elif similarity > 0.50:
        print("🟠 SUSPICIOUS")
        print("   ⚠️  Somewhat different from training patterns")
        malicious_count += 1
    else:
        print("🔴 LIKELY MALICIOUS")
        print("   🚨 Very different from training patterns")
        malicious_count += 1
    print("-" * 80)

print(f"\n📊 Summary:")
print(f"🟢 URLs similar to benign training: {benign_count}")
print(f"🔴 URLs different from benign training: {malicious_count}")
print(f"📋 Total URLs analyzed: {len(test_urls)}")

print(f"\n🎯 Interpretation:")
print(f"• Reconstruction Error: How different the URL is from what the model expects")
print(f"• Lower error = More similar to {config['max_samples']} benign training URLs")  
print(f"• Higher error = More different from training (potentially malicious)")
print(f"• Your training loss was ~0.043, so errors around 0.04-0.06 are normal for benign URLs")
