import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import json
import os
import numpy as np

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
        
        for param in list(self.cysec.encoder.layer[:6].parameters()):
            param.requires_grad = False
        for param in list(self.electra.encoder.layer[:6].parameters()):
            param.requires_grad = False

    def forward(self, cysec_ids, cysec_mask, electra_ids, electra_mask):
        cysec_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:, 0, :]
        electra_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:, 0, :]
        combined = torch.cat((cysec_out, electra_out), dim=1)
        return self.classifier(combined)

def load_model():
    model_path = "cysec_electra_oneclass_model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model path not found: {model_path}")
    
    with open(f"{model_path}/training_config.json", "r") as f:
        config = json.load(f)
    
    cysec_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/cysec_tokenizer")
    electra_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/electra_tokenizer")
    
    model = CySecElectraFusion(config["cysecbert_model"], config["electra_model"], dropout_rate=config["dropout_rate"])
    
    weights_path = "cysec_electra_oneclass.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Model weights not found at: {weights_path}")
    
    state_dict = torch.load(weights_path, map_location="cpu")
    if "fusion_encoder" in state_dict and "autoencoder" in state_dict:
        print("⚠️ Detected autoencoder checkpoint — using fusion_encoder weights for inference.")
        model_state = state_dict["fusion_encoder"]
        model.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, cysec_tokenizer, electra_tokenizer, config, device

def classify_urls(urls, model, cysec_tokenizer, electra_tokenizer, config, device):
    cysec_enc = cysec_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
    electra_enc = electra_tokenizer(urls, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
    
    cysec_ids = cysec_enc["input_ids"].to(device)
    cysec_mask = cysec_enc["attention_mask"].to(device)
    electra_ids = electra_enc["input_ids"].to(device)
    electra_mask = electra_enc["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(cysec_ids, cysec_mask, electra_ids, electra_mask)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    results = []
    for url, pred, prob in zip(urls, preds.cpu().numpy(), probs.cpu().numpy()):
        classification = "BENIGN" if pred == 0 else "MALICIOUS"
        results.append({
            "url": url,
            "classification": classification,
            "confidence": float(prob[pred]),
            "benign_prob": float(prob[0]),
            "malicious_prob": float(prob[1])
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
    model, cysec_tokenizer, electra_tokenizer, config, device = load_model()
    print(f"✅ Model loaded successfully on: {device}\n")
    
    print("🔍 Analyzing URLs...\n")
    results = classify_urls(test_urls, model, cysec_tokenizer, electra_tokenizer, config, device)
    
    print("="*100)
    print(f"{'URL':<65} {'CLASSIFICATION':<15} {'CONFIDENCE':<10} {'BENIGN':<10} {'MALICIOUS':<10}")
    print("="*100)
    
    benign_count = sum(r["classification"] == "BENIGN" for r in results)
    malicious_count = len(results) - benign_count
    
    for r in results:
        url = r["url"][:62] + "..." if len(r["url"]) > 65 else r["url"]
        icon = "🟢" if r["classification"] == "BENIGN" else "🔴"
        print(f"{icon} {url:<63} {r['classification']:<15} {r['confidence']:.4f}   {r['benign_prob']:.4f}     {r['malicious_prob']:.4f}")
    
    print("="*100)
    print(f"\nSummary:")
    print(f"   🟢 Benign URLs: {benign_count}")
    print(f"   🔴 Malicious URLs: {malicious_count}")
    print(f"   📋 Total Analyzed: {len(test_urls)}")

if __name__ == "__main__":
    main()
