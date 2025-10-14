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

model_path = "cysec_electra_oneclass_model"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model path not found: {model_path}")

with open(f"{model_path}/training_config.json", "r") as f:
    config = json.load(f)

cysec_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/cysec_tokenizer")
electra_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/electra_tokenizer")

fusion_encoder = FusionEncoder(config["cysecbert_model"], config["electra_model"])

weights_path = "cysec_electra_oneclass.pth"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"❌ Model weights not found at: {weights_path}")

state_dict = torch.load(weights_path, map_location="cpu")
fusion_encoder.load_state_dict(state_dict["fusion_encoder"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_encoder.to(device)
fusion_encoder.eval()

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

print("🔮 Analyzing URL deviations from benign patterns...")

with torch.no_grad():
    embeddings = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
    
    # Calculate how "weird" each URL embedding is compared to expected benign patterns
    # Higher values = more deviation from learned benign patterns
    deviations = torch.norm(embeddings, dim=1).cpu().numpy()

print("\n" + "="*80)
print("🔍 URL Deviation Analysis from Benign Training")
print("="*80)

# Normalize deviations to 0-1 scale for easier interpretation
max_dev = max(deviations)
normalized_deviations = [dev / max_dev for dev in deviations]

benign_count = 0
malicious_count = 0

for url, deviation, norm_dev in zip(test_urls, deviations, normalized_deviations):
    print(f"URL: {url}")
    print(f"Raw Deviation: {deviation:.4f}")
    print(f"Normalized Deviation: {norm_dev:.4f}")
    
    # Convert deviation to "benign confidence" (inverse relationship)
    benign_confidence = max(0, 1.0 - norm_dev)
    
    if norm_dev < 0.3:
        print(f"🟢 BENIGN - Confidence: {benign_confidence:.2%}")
        print("   ✓ Closely matches learned benign patterns")
        benign_count += 1
    elif norm_dev < 0.6:
        print(f"🟡 SUSPICIOUS - Confidence: {benign_confidence:.2%}")
        print("   ⚠️  Somewhat different from benign patterns")
        malicious_count += 1
    else:
        print(f"🔴 MALICIOUS - Confidence: {benign_confidence:.2%}")
        print("   🚨 Significantly different from benign patterns")
        malicious_count += 1
    print("-" * 80)

print(f"\n📊 Summary:")
print(f"🟢 URLs matching benign patterns: {benign_count}")
print(f"🔴 URLs deviating from benign patterns: {malicious_count}")
print(f"📋 Total URLs analyzed: {len(test_urls)}")

print(f"\n🎯 Interpretation:")
print(f"• Lower deviation = More similar to training benign URLs")
print(f"• Higher deviation = More different from training benign URLs")
print(f"• Model was trained on {config['max_samples']} benign URLs")
print(f"• High deviation suggests potential malicious content")
