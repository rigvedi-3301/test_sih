import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_path = "distilbert_url_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_urls = [
    "http://example.com",
    "http://malicious-site.com/bad",
    "https://safe-site.org",
    "https://github.com"
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
