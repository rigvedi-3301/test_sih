import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wandb
from torch.cuda.amp import autocast, GradScaler

wandb.login()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(42)

hyperparams = {
    "warmup_ratio": 0.08,
    "model_name": "distilbert-base-uncased",
    "max_length": 256,
    "batch_size": 64,  
    "learning_rate": 1.5e-5,
    "epochs": 3,
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss",
    "train_val_split": 0.8,
    "shuffle_data": True,
    "scheduler_type": "cosine",
    "subset_size": 200000  
}

wandb.init(project="url_malware_demo", name="test_run_full_ft_1", config=hyperparams)

df = pd.read_csv("./minitrain_data/kaggle_demo.csv")

if hyperparams["subset_size"] is not None and hyperparams["subset_size"] < len(df):
    df = df.sample(n=hyperparams["subset_size"], random_state=42).reset_index(drop=True)
else:
    train_df, val_df = train_test_split(
    df,
    test_size=1 - hyperparams["train_val_split"],
    stratify=df["result"],
    random_state=42
)


print(f"Using dataset size: {len(df)}")

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
labels = torch.tensor(df["result"].values).to(torch.long)

train_idx, val_idx = train_test_split(
    range(len(df)),
    test_size=1 - hyperparams["train_val_split"],
    stratify=labels.numpy(),
    random_state=42
)

train_dataset = TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx])
val_dataset = TensorDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx])

print(f"Train size: {len(train_dataset)} | Validation size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"], num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

scaler = GradScaler()

for epoch in range(hyperparams["epochs"]):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        b_input_ids, b_attention, b_labels = [x.to(device) for x in batch]
        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention)
            loss = loss_fn(outputs.logits, b_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

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

    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Detection Rate: {detection_rate:.4f}")
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
