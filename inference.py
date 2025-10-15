import torch
import os
import json
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel

# -------------------------------
# Fusion Encoder
# -------------------------------
class FusionEncoder(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name, freeze_layers=8):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        self.out_dim = self.cysec.config.hidden_size + self.electra.config.hidden_size

        # Freeze initial layers if specified
        for param in list(self.cysec.encoder.layer[:freeze_layers].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:freeze_layers].parameters()):
            param.requires_grad = False

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cy_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:, 0, :]
        el_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:, 0, :]
        return torch.cat((cy_out, el_out), dim=1)


# -------------------------------
# Autoencoder
# -------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z


# -------------------------------
# Load Model
# -------------------------------
def load_model(model_path="cysec_electra_oneclass_model_v4", weights_path="cysec_electra_oneclass_v4.pth"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model path not found: {model_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Weights not found: {weights_path}")

    # Load training config
    with open(os.path.join(model_path, "training_config.json"), "r") as f:
        config = json.load(f)

    cysec_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "cysec_tokenizer"))
    electra_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "electra_tokenizer"))

    fusion_encoder = FusionEncoder(config["cysecbert_model"], config["electra_model"], freeze_layers=config.get("freeze_layers", 8))
    autoencoder = AutoEncoder(fusion_encoder.out_dim, dropout_rate=config.get("dropout_rate", 0.5))

    # Load saved weights
    state_dict = torch.load(weights_path, map_location="cpu")
    fusion_encoder.load_state_dict(state_dict["fusion_encoder"])
    autoencoder.load_state_dict(state_dict["autoencoder"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_encoder.to(device).eval()
    autoencoder.to(device).eval()

    return fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device


# -------------------------------
# URL Classification
# -------------------------------
def classify_urls(urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device, threshold=None):
    cysec_enc = cysec_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
    electra_enc = electra_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")

    cysec_ids, cysec_mask = cysec_enc["input_ids"].to(device), cysec_enc["attention_mask"].to(device)
    electra_ids, electra_mask = electra_enc["input_ids"].to(device), electra_enc["attention_mask"].to(device)

    with torch.no_grad():
        embeddings = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
        reconstructed, _ = autoencoder(embeddings)
        errors = torch.mean((embeddings - reconstructed) ** 2, dim=1).cpu().numpy()

    # Adaptive threshold if none provided
    if threshold is None:
        threshold = np.mean(errors) + 0.1 * np.std(errors)

    results = []
    for url, error in zip(urls, errors):
        results.append({
            "url": url,
            "classification": "BENIGN" if error <= threshold else "MALICIOUS"
        })
    return results, threshold


# -------------------------------
# Main
# -------------------------------
def main():
    test_urls = [
        "https://shop.example.com/product/12345?ref=google&utm_source=email",
        "https://docs.example.org/user-guide/v2.1/user_guide.pdf",
        "http://192.0.2.45/downloads/update.exe",
        "http://203.0.113.77/installer/latest_installer.zip?payload=cmd.exe",
        "https://login.example.com/?user=admin&pass=%3Cscript%3Ealert(1)%3C%2Fscript%3E",
        "https://www.bankofamerica.com/login",
        "http://bit.ly/2FakeLink",
    ]

    print("🔄 Loading model...")
    fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device = load_model()

    results, threshold = classify_urls(test_urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device)

    print(f"\nAdaptive Threshold: {threshold:.6f}\n")
    print("="*95)
    print(f"{'URL':<80} {'CLASS':<12}")
    print("="*95)
    for r in results:
        icon = "🟢" if r["classification"] == "BENIGN" else "🔴"
        url = r["url"][:77] + "..." if len(r["url"]) > 80 else r["url"]
        print(f"{icon} {url:<79} {r['classification']:<12}")
    print("="*95)


if __name__ == "__main__":
    main()
