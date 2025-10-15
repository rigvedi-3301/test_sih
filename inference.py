import os
import json
import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel

# ===========================
# MODEL DEFINITIONS
# ===========================
class FusionEncoder(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name, freeze_layers=8):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        self.out_dim = self.cysec.config.hidden_size + self.electra.config.hidden_size
        # Freeze first few layers
        for param in list(self.cysec.encoder.layer[:freeze_layers].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:freeze_layers].parameters()):
            param.requires_grad = False

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cy_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:, 0, :]
        el_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:, 0, :]
        return torch.cat((cy_out, el_out), dim=1)

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
            nn.Dropout(dropout_rate*0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate*0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate*0.5),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

# ===========================
# LOAD MODEL
# ===========================
def load_model():
    model_path = "cysec_electra_oneclass_model_v4"
    weights_path = "cysec_electra_oneclass_v4.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model path not found: {model_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Weights not found: {weights_path}")

    with open(f"{model_path}/training_config.json", "r") as f:
        config = json.load(f)

    cysec_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/cysec_tokenizer")
    electra_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/electra_tokenizer")

    fusion_encoder = FusionEncoder(config["cysecbert_model"], config["electra_model"], freeze_layers=config.get("freeze_layers", 8))
    autoencoder = AutoEncoder(fusion_encoder.out_dim, dropout_rate=config.get("dropout_rate", 0.5))

    state_dict = torch.load(weights_path, map_location="cpu")
    fusion_encoder.load_state_dict(state_dict["fusion_encoder"])
    autoencoder.load_state_dict(state_dict["autoencoder"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_encoder.to(device).eval()
    autoencoder.to(device).eval()

    return fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device

# ===========================
# CLASSIFY URLS
# ===========================
def classify_urls(urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device, calibration_urls=None):
    # Encode test URLs
    cysec_enc = cysec_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
    electra_enc = electra_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
    cysec_ids, cysec_mask = cysec_enc["input_ids"].to(device), cysec_enc["attention_mask"].to(device)
    electra_ids, electra_mask = electra_enc["input_ids"].to(device), electra_enc["attention_mask"].to(device)

    with torch.no_grad():
        embeddings = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
        reconstructed, _ = autoencoder(embeddings)
        errors = torch.mean((embeddings - reconstructed)**2, dim=1).cpu().numpy()

    # Optionally include calibration URLs (malicious) to set threshold
    if calibration_urls:
        cysec_cal = cysec_tokenizer(calibration_urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
        electra_cal = electra_tokenizer(calibration_urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
        cysec_ids_cal, cysec_mask_cal = cysec_cal["input_ids"].to(device), cysec_cal["attention_mask"].to(device)
        electra_ids_cal, electra_mask_cal = electra_cal["input_ids"].to(device), electra_cal["attention_mask"].to(device)
        with torch.no_grad():
            emb_cal = fusion_encoder(cysec_ids_cal, cysec_mask_cal, electra_ids_cal, electra_mask_cal)
            rec_cal, _ = autoencoder(emb_cal)
            errors_cal = torch.mean((emb_cal - rec_cal)**2, dim=1).cpu().numpy()
        combined_errors = np.concatenate([errors, errors_cal])
    else:
        combined_errors = errors

    # Adaptive threshold: mean + 3*std
    mean_err = np.mean(combined_errors)
    std_err = np.std(combined_errors)
    threshold = mean_err + 3 * std_err

    # Classification
    results = []
    for url, error in zip(urls, errors):
        classification = "BENIGN" if error <= threshold else "MALICIOUS"
        results.append({"url": url, "classification": classification, "reconstruction_error": float(error)})

    return results, threshold, errors

# ===========================
# MAIN
# ===========================
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

    calibration_urls = [
        "http://malware.com/badfile.exe",
        "http://phishing.example.com/login",
        "http://example.com/admin/delete_all",
        "http://dangerous-site.ru/download",
        "http://evil.com/?cmd=rm -rf /"
    ]

    print("🔄 Loading model...")
    fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device = load_model()

    results, threshold, errors = classify_urls(
        test_urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device, calibration_urls=calibration_urls
    )

    print("\n📊 Reconstruction Error Distribution:")
    print(f"Min: {np.min(errors):.6f}, Max: {np.max(errors):.6f}, Mean: {np.mean(errors):.6f}")
    print(f"Adaptive Threshold (mean+3*std): {threshold:.6f}\n")

    print("="*110)
    print(f"{'URL':<80} {'CLASS':<12} {'ERROR':<10}")
    print("="*110)
    for r in results:
        icon = "🟢" if r["classification"]=="BENIGN" else "🔴"
        url = r["url"][:77]+"..." if len(r["url"])>80 else r["url"]
        print(f"{icon} {url:<79} {r['classification']:<12} {r['reconstruction_error']:.6f}")
    print("="*110)

if __name__ == "__main__":
    main()
