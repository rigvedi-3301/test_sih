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
        # Get embeddings (detached to match training behavior)
        embeddings = fusion_encoder(cysec_ids, cysec_mask, electra_ids, electra_mask).detach()
        
        # Reconstruct using autoencoder
        reconstructed, _ = autoencoder(embeddings)
        
        # Calculate reconstruction error
        reconstruction_errors = torch.mean((embeddings - reconstructed) ** 2, dim=1).cpu().numpy()
    
    # If no threshold provided, use a conservative approach based on error distribution
    # We'll identify likely benign URLs first, then set threshold
    if threshold is None:
        # Sort errors and use the gap detection method
        sorted_errors = np.sort(reconstruction_errors)
        
        # Find the largest gap in the sorted errors (indicates separation between benign/malicious)
        gaps = np.diff(sorted_errors)
        
        if len(gaps) > 0:
            # Find where the largest gap occurs
            largest_gap_idx = np.argmax(gaps)
            
            # Set threshold at the midpoint of the largest gap
            threshold = (sorted_errors[largest_gap_idx] + sorted_errors[largest_gap_idx + 1]) / 2
            
            # Safety check: threshold should be reasonable relative to min error
            min_error = sorted_errors[0]
            if threshold < min_error * 1.5:
                # If threshold is too close to minimum, use a more conservative multiplier
                threshold = min_error * 2.0
        else:
            # Fallback: use mean of lowest errors
            threshold = np.mean(sorted_errors[:max(1, len(sorted_errors)//3)]) * 2.5
    
    results = []
    for url, error in zip(urls, reconstruction_errors):
        # Calculate how far the error is from threshold
        error_ratio = error / threshold
        
        if error <= threshold:
            classification = "BENIGN"
            # Confidence based on how much below threshold
            confidence = max(0.5, 1.0 - error_ratio)
        else:
            classification = "MALICIOUS"
            # Confidence based on how much above threshold
            confidence = min(0.99, 0.5 + (error_ratio - 1.0) * 0.5)
        
        results.append({
            "url": url,
            "classification": classification,
            "confidence": confidence,
            "reconstruction_error": error,
            "error_ratio": error_ratio
        })
    
    return results, threshold

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
    results, threshold = classify_urls(test_urls, fusion_encoder, autoencoder, cysec_tokenizer, electra_tokenizer, config, device)
    
    # Display results
    print("="*110)
    print(f"{'URL':<60} {'CLASS':<12} {'CONFIDENCE':<12} {'ERROR':<10} {'RATIO':<8}")
    print("="*110)
    
    benign_count = 0
    malicious_count = 0
    
    for result in results:
        url = result["url"][:57] + "..." if len(result["url"]) > 60 else result["url"]
        classification = result["classification"]
        confidence = f"{result['confidence']:.2%}"
        error = f"{result['reconstruction_error']:.4f}"
        ratio = f"{result['error_ratio']:.2f}x"
        
        if classification == "BENIGN":
            icon = "🟢"
            benign_count += 1
        else:
            icon = "🔴"
            malicious_count += 1
        
        print(f"{icon} {url:<58} {classification:<12} {confidence:<12} {error:<10} {ratio:<8}")
    
    print("="*110)
    print(f"\n📊 Summary:")
    print(f"   🟢 Benign URLs: {benign_count}")
    print(f"   🔴 Malicious URLs: {malicious_count}")
    print(f"   📋 Total Analyzed: {len(test_urls)}")
    print(f"   🎯 Threshold: {threshold:.6f}")
    print(f"\n💡 Interpretation:")
    print(f"   • Ratio < 1.0x → BENIGN (below threshold)")
    print(f"   • Ratio > 1.0x → MALICIOUS (above threshold)")
    print(f"   • Threshold found using gap detection in error distribution")
    print(f"\n⚙️  To adjust sensitivity:")
    print(f"   • If too many false positives: increase threshold manually")
    print(f"   • If too many false negatives: decrease threshold manually")
    print(f"   • Call classify_urls(..., threshold=YOUR_VALUE) to set manually")

if __name__ == "__main__":
    main()
