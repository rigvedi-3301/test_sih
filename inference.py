import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import json
import os
import numpy as np

# ============================================================
#  CySec + ELECTRA Fusion Model Definition
# ============================================================
class CySecElectraFusion(nn.Module):
    def __init__(self, cysec_model_name, electra_model_name, dropout_rate=0.55):
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

        # Freeze early encoder layers to save GPU memory
        for param in list(self.cysec.encoder.layer[:6].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:6].parameters()):
            param.requires_grad = False

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cysec_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:, 0, :]
        electra_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:, 0, :]
        combined = torch.cat((cysec_out, electra_out), dim=1)
        return self.classifier(combined)


# ============================================================
#  Load Model + Tokenizers
# ============================================================
def load_model():
    model_path = "cysec_electra_oneclass_model"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    with open(f"{model_path}/training_config.json", "r") as f:
        config = json.load(f)

    cysec_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/cysec_tokenizer")
    electra_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/electra_tokenizer")

    model = CySecElectraFusion(config["cysecbert_model"], config["electra_model"], dropout_rate=config["dropout_rate"])

    weights_path = f"{model_path}/pytorch_model.bin"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")

    # Detect autoencoder fusion model (used in one-class setup)
    if "fusion_encoder" in state_dict:
        print("⚠️ Detected autoencoder checkpoint — using fusion_encoder weights for inference.")
        state_dict = {k.replace("fusion_encoder.", ""): v for k, v in state_dict.items() if "fusion_encoder." in k}

    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("✅ Model loaded successfully on:", device)
    return model, cysec_tokenizer, electra_tokenizer, config, device


# ============================================================
#  Inference / Classification
# ============================================================
def classify_urls(urls, model, cysec_tokenizer, electra_tokenizer, config, device, benign_threshold=0.45):
    cysec_encodings = cysec_tokenizer(
        urls,
        padding=True,
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt"
    )
    electra_encodings = electra_tokenizer(
        urls,
        padding=True,
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt"
    )

    cysec_ids = cysec_encodings["input_ids"].to(device)
    cysec_mask = cysec_encodings["attention_mask"].to(device)
    electra_ids = electra_encodings["input_ids"].to(device)
    electra_mask = electra_encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(cysec_ids, cysec_mask, electra_ids, electra_mask)
        probs = torch.softmax(logits, dim=1)
        benign_probs = probs[:, 0]  # class 0 = benign

    results = []
    for url, benign_prob in zip(urls, benign_probs.cpu().numpy()):
        if benign_prob < benign_threshold:
            classification = "MALICIOUS"
            confidence = 1 - benign_prob
        else:
            classification = "BENIGN"
            confidence = benign_prob

        results.append({
            "url": url,
            "classification": classification,
            "confidence": float(confidence),
            "benign_prob": float(benign_prob),
            "malicious_prob": float(1 - benign_prob)
        })

    return results


# ============================================================
#  MAIN SCRIPT
# ============================================================
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
    model, cysec_tokenizer, electra_tokenizer, config, device = load_model()
    print("\n🔍 Analyzing URLs...\n")

    results = classify_urls(test_urls, model, cysec_tokenizer, electra_tokenizer, config, device, benign_threshold=0.45)

    print("="*100)
    print(f"{'URL':<65} {'CLASSIFICATION':<15} {'CONFIDENCE':<10} {'BENIGN':<10} {'MALICIOUS':<10}")
    print("="*100)

    benign_count = 0
    malicious_count = 0

    for result in results:
        url = result["url"][:62] + "..." if len(result["url"]) > 65 else result["url"]
        classification = result["classification"]
        confidence = f"{result['confidence']:.4f}"
        benign_prob = f"{result['benign_prob']:.4f}"
        malicious_prob = f"{result['malicious_prob']:.4f}"

        if classification == "BENIGN":
            icon = "🟢"
            benign_count += 1
        else:
            icon = "🔴"
            malicious_count += 1

        print(f"{icon} {url:<63} {classification:<15} {confidence:<10} {benign_prob:<10} {malicious_prob:<10}")

    print("="*100)
    print(f"\nSummary:")
    print(f"   🟢 Benign URLs: {benign_count}")
    print(f"   🔴 Malicious URLs: {malicious_count}")
    print(f"   📋 Total Analyzed: {len(test_urls)}")


if __name__ == "__main__":
    main()
