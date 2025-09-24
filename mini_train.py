import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from sklearn.metrics import accuracy_score
import wandb
from torch.cuda.amp import autocast, GradScaler

wandb.login()

hyperparams = {
    "warmup_ratio": 0.1,
    "model_name": "distilbert-base-uncased",
    "max_length": 256,
    "batch_size": 32,  
    "learning_rate": 2e-5,
    "epochs": 5,
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss",
    "train_val_split": 0.8,
    "shuffle_data": True,
    "scheduler_type": "cosine"
}

wandb.init(project="url_malware_demo", name="test_run_full_ft_1", config=hyperparams)

df = pd.read_csv("./minitrain_data/kaggle_demo.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

tokenizer = DistilBertTokenizer.from_pretrained(hyperparams["model_name"])
encoding = tokenizer(
    list(df["url"]),
    padding=True,
    truncation=True,
    max_length=hyperparams["max_length"],
    return_tensors="pt"
)

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
labels = torch.tensor(df["result"].values)


train_size = int(hyperparams["train_val_split"] * len(df))
val_size = len(df) - train_size
dataset = TensorDataset(input_ids, attention_mask, labels)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained(
    hyperparams["model_name"],
    num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"])
loss_fn = torch.nn.CrossEntropyLoss()

total_steps = len(train_loader) * hyperparams["epochs"]
warmup_steps = int(hyperparams["warmup_ratio"] * total_steps)

scheduler = get_scheduler(
    name=hyperparams["scheduler_type"],
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


scaler = GradScaler(device_type="cuda)

for epoch in range(hyperparams["epochs"]):
    model.train()
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_attention, b_labels = [x.to(device) for x in batch]
        optimizer.zero_grad()

        with autocast(device_type="cuda):  
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention)
            loss = loss_fn(outputs.logits, b_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_attention, b_labels = [x.to(device) for x in batch]
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)

    malicious_idx = [i for i, lbl in enumerate(all_labels) if lbl == 1]
    true_positives = sum(all_preds[i] == 1 for i in malicious_idx)
    total_malicious = len(malicious_idx)
    detection_rate = true_positives / total_malicious if total_malicious > 0 else 0.0

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Detection Rate: {detection_rate:.4f}")
    wandb.log({
        "epoch": epoch+1,
        "loss": avg_loss,
        "val_accuracy": val_acc,
        "detection_rate": detection_rate,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

model.save_pretrained("distilbert_url_model")
tokenizer.save_pretrained("distilbert_url_model")
wandb.save("distilbert_url_model/*")

print("Training complete! Model saved as 'distilbert_url_model'.")
