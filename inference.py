import os
import json
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

class FusionEncoder(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        self.out_dim = self.cysec.config.hidden_size + self.electra.config.hidden_size

        for param in list(self.cysec.encoder.layer[:6].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:6].parameters()):
            param.requires_grad = False

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cy_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:, 0, :]
        el_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:, 0, :]
        return torch.cat((cy_out, el_out), dim=1)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

def load_model():
    model_path = "cysec_electra_oneclass_model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    with open(f"{model_path}/training_config.json", "r") as f:
        config = json.load(f)

    cysec_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/cysec_tokenizer")
    electra_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/electra_tokenizer")

    fusion_encoder = FusionEncoder(config["cysecbert_model"], config["electra_model"]).to("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = AutoEncoder(fusion_encoder.out_dim, dropout_rate=config["dropout_rate"]).to("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = "cysec_electra_oneclass.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    fusion_encoder.load_state_dict(checkpoint["fusion_encoder"])
    autoencoder.load_state_dict(checkpoint["autoencoder"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_encoder.eval()
    autoencoder.eval()

    return fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device

def classify_urls(urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device, threshold=0.015):
    cysec_encodings = cysec_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
    electra_encodings = electra_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")

    cysec_ids = cysec_encodings["input_ids"].to(device)
    cysec_mask = cysec_encodings["attention_mask"].to(device)
    electra_ids = electra_encodings["input_ids"].to(device)
    electra_mask = electra_encodings["attention_mask"].to(device)

    results = []
    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        fused = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
        reconstructed, _ = autoencoder(fused)
        losses = criterion(reconstructed, fused).mean(dim=1).cpu().numpy()

    for url, loss in zip(urls, losses):
        classification = "BENIGN" if loss <= threshold else "MALICIOUS"
        results.append({
            "url": url,
            "reconstruction_loss": loss,
            "classification": classification
        })

    return results

def main():
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

    print("Loading model...")
    fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device = load_model()
    print(f"✅ Model loaded successfully on: {device}\n")

    print("🔍 Analyzing URLs...\n")
    results = classify_urls(test_urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device)

    print("=" * 100)
    print(f"{'URL':<65} {'CLASSIFICATION':<15} {'LOSS':<15}")
    print("=" * 100)

    benign_count, malicious_count = 0, 0

    for r in results:
        url = r["url"][:62] + "..." if len(r["url"]) > 65 else r["url"]
        loss_str = f"{r['reconstruction_loss']:.6f}"
        if r["classification"] == "BENIGN":
            benign_count += 1
            icon = "🟢"
        else:
            malicious_count += 1
            icon = "🔴"
        print(f"{icon} {url:<63} {r['classification']:<15} {loss_str:<15}")

    print("=" * 100)
    print(f"\nSummary:")
    print(f"   🟢 Benign URLs: {benign_count}")
    print(f"   🔴 Malicious URLs: {malicious_count}")
    print(f"   📋 Total Analyzed: {len(test_urls)}")

if __name__ == "__main__":
    main()
