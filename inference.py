import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

model_path = "distilbert_url_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_urls = [
    "https://www.example.com/",
    "https://shop.example.com/product/12345?ref=google&utm_source=email",
    "https://docs.example.org/user-guide/v2.1/user_guide.pdf",
    "https://accounts.example.com/login?continue=/dashboard",
    "http://192.0.2.45/downloads/update.exe",
    "http://203.0.113.77/installer/latest_installer.zip?payload=cmd.exe",
    "https://www.example.com/%2e%2e/%2e%2e/admin/config.php",
    "https://login.example.com/?user=admin&pass=%3Cscript%3Ealert(1)%3C%2Fscript%3E"
]

encoding = tokenizer(
    test_urls,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

label_map = {0: "benign", 1: "malicious"}

for url, pred, prob in zip(test_urls, preds.cpu().numpy(), probs.cpu().numpy()):
    print(f"URL: {url}")
    print(f"Prediction: {label_map[pred]} | Probabilities: benign={prob[0]:.4f}, malicious={prob[1]:.4f}")
    print("-" * 50)
