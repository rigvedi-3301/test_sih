import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json
import os

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
        cysec_out = self.cysec(input_ids=cysec_ids, attention_mask=cysec_mask).last_hidden_state[:,0,:]
        electra_out = self.electra(input_ids=electra_ids, attention_mask=electra_mask).last_hidden_state[:,0,:]
        combined = torch.cat((cysec_out, electra_out), dim=1)
        return self.classifier(combined)

model_path = "cysec_electra_fusion_model_benign_250k"
with open(f"{model_path}/training_config.json", "r") as f:
    config = json.load(f)

cysec_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/cysec_tokenizer")
electra_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/electra_tokenizer")

model = CySecElectraFusion(
    config["cysecbert_model"],
    config["electra_model"],
    dropout_rate=config["dropout_rate"]
)

model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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

print("🔮 Running inference...")

with torch.no_grad():
    logits = model(cysec_ids, cysec_mask, electra_ids, electra_mask)
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

label_map = {0: "benign", 1: "malicious"}

print("\n" + "="*80)
print("🔒 CySecBERT + ELECTRA Fusion Model Results")
print("="*80)

for url, pred, prob in zip(test_urls, preds.cpu().numpy(), probs.cpu().numpy()):
    confidence = prob[pred]
    print(f"URL: {url}")
    print(f"Prediction: {label_map[pred]} | Confidence: {confidence:.4f}")
    print(f"Probabilities: benign={prob[0]:.4f}, malicious={prob[1]:.4f}")
    
    if confidence < 0.7:
        print("⚠️  Low confidence prediction")
    elif confidence > 0.95:
        print("✅ High confidence prediction")
    
    print("-" * 80)

benign_count = (preds == 0).sum().item()
malicious_count = (preds == 1).sum().item()

print(f"\n📊 Summary:")
print(f"Benign URLs: {benign_count}")
print(f"Malicious URLs: {malicious_count}")
print(f"Total URLs analyzed: {len(test_urls)}")

print(f"\n💡 Note: This model was trained on {config['max_samples']} benign-only samples")
print("   It may be biased towards predicting 'benign' due to the training data")
