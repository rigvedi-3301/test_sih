import torch, os, json, numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel

# -------------------
# Fusion Encoder
# -------------------
class FusionEncoder(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name):
        super().__init__()
        self.cysec = AutoModel.from_pretrained(cysec_model_name)
        self.electra = AutoModel.from_pretrained(electra_model_name)
        self.out_dim = self.cysec.config.hidden_size + self.electra.config.hidden_size

        # freeze first 8 layers
        for param in list(self.cysec.encoder.layer[:8].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:8].parameters()):
            param.requires_grad = False

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cy_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:, 0, :]
        el_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:, 0, :]
        return torch.cat((cy_out, el_out), dim=1)

# -------------------
# AutoEncoder (matches v4 checkpoint)
# -------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

# -------------------
# Load Model
# -------------------
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

    fusion_encoder = FusionEncoder(config["cysecbert_model"], config["electra_model"])
    autoencoder = AutoEncoder(fusion_encoder.out_dim, dropout_rate=config.get("dropout_rate", 0.5))

    state_dict = torch.load(weights_path, map_location="cpu")
    fusion_encoder.load_state_dict(state_dict["fusion_encoder"])
    autoencoder.load_state_dict(state_dict["autoencoder"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_encoder.to(device).eval()
    autoencoder.to(device).eval()

    return fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device

# -------------------
# URL Classification
# -------------------
def classify_urls(urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device, threshold=None):
    cysec_enc = cysec_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
    electra_enc = electra_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")

    cysec_ids, cysec_mask = cysec_enc["input_ids"].to(device), cysec_enc["attention_mask"].to(device)
    electra_ids, electra_mask = electra_enc["input_ids"].to(device), electra_enc["attention_mask"].to(device)

    with torch.no_grad():
        embeddings = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
        reconstructed, _ = autoencoder(embeddings)
        errors = torch.mean((embeddings - reconstructed) ** 2, dim=1).cpu().numpy()

    if threshold is None:
        threshold = 0.15

    results = []
    for url, error in zip(urls, errors):
        results.append({
            "url": url,
            "classification": "BENIGN" if error <= threshold else "MALICIOUS",
            "reconstruction_error": float(error)
        })
    return results, threshold, errors

# -------------------
# Main
# -------------------
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

    print("🔄 Loading model...")
    fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device = load_model()

    results, threshold, errors = classify_urls(test_urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device)

    print("\n📊 Reconstruction Error Distribution:")
    print(f"Min: {np.min(errors):.6f}, Max: {np.max(errors):.6f}, Mean: {np.mean(errors):.6f}")
    print(f"Threshold: {threshold:.6f}\n")

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
