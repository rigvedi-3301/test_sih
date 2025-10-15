import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
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

def load_model():
    """Load the trained model and tokenizers"""
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
    
    return fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device

def classify_urls(urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device, threshold=None):
    """Classify URLs as malicious or benign"""
    
    # Tokenize URLs
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
        # Get embeddings (no detach - consistent with fixed training)
        embeddings = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask)
        
        # Reconstruct using autoencoder
        reconstructed, _ = autoencoder(embeddings)
        
        # Calculate reconstruction error
        reconstruction_errors = torch.mean((embeddings - reconstructed) ** 2, dim=1).cpu().numpy()
    
    # Use a fixed threshold based on typical training loss
    # Since your training loss was ~0.043 (validation), we set threshold slightly higher
    # URLs significantly above this are anomalous (malicious)
    if threshold is None:
        # Conservative threshold: 3x the expected training loss
        # This captures natural variance while flagging true outliers
        threshold = 0.15  # Adjust this value based on your validation loss
    
    results = []
    for url, error in zip(urls, reconstruction_errors):        
        if error <= threshold:
            classification = "BENIGN"
        else:
            classification = "MALICIOUS"
        
        results.append({
            "url": url,
            "classification": classification,
            "reconstruction_error": error
        })
    
    return results, threshold, reconstruction_errors

def main():
    # Test URLs - mix of clearly benign and clearly malicious
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
    print(f"✅ Model loaded successfully on: {device}\n")
    
    print("🔍 Analyzing URLs...\n")
    results, threshold, all_errors = classify_urls(test_urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device)
    
    # Print error distribution first for debugging
    print("📊 Reconstruction Error Distribution:")
    print(f"   Min Error:  {np.min(all_errors):.6f}")
    print(f"   Max Error:  {np.max(all_errors):.6f}")
    print(f"   Mean Error: {np.mean(all_errors):.6f}")
    print(f"   Std Error:  {np.std(all_errors):.6f}")
    print(f"   Threshold:  {threshold:.6f}\n")
    
    # Display results
    print("="*90)
    print(f"{'URL':<65} {'CLASSIFICATION':<15} {'ERROR':<10}")
    print("="*90)
    
    benign_count = 0
    malicious_count = 0
    
    for result in results:
        url = result["url"][:62] + "..." if len(result["url"]) > 65 else result["url"]
        classification = result["classification"]
        error = f"{result['reconstruction_error']:.6f}"
        
        if classification == "BENIGN":
            icon = "🟢"
            benign_count += 1
        else:
            icon = "🔴"
            malicious_count += 1
        
        print(f"{icon} {url:<63} {classification:<15} {error:<10}")
    
    print("="*90)
    print(f"\n📊 Summary:")
    print(f"   🟢 Benign URLs: {benign_count}")
    print(f"   🔴 Malicious URLs: {malicious_count}")
    print(f"   📋 Total Analyzed: {len(test_urls)}")
    print(f"\n💡 Current threshold: {threshold:.6f}")
    print(f"   • To make MORE sensitive (catch more malicious): DECREASE threshold")
    print(f"   • To make LESS sensitive (fewer false alarms): INCREASE threshold")
    print(f"   • Suggested range based on your data: 0.05 to 0.30")

if __name__ == "__main__":
    main()
