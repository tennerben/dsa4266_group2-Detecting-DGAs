import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import CanineTokenizer, CanineForSequenceClassification


def train(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs, writer, save_path):
    best_loss = np.inf

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for input_ids, attention_mask, labels in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item()
            out.loss.backward()
            optimizer.step()
            scheduler.step()

        writer.add_scalar("Loss/train", total_loss, epoch)

        model.eval()
        total_loss = 0
        all_preds = np.array([])
        all_labels = np.array([])

        for input_ids, attention_mask, labels in val_dataloader:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item()
            all_preds = np.append(all_preds, out.logits.argmax(1).cpu().numpy())
            all_labels = np.append(all_labels, labels.cpu())

        writer.add_scalar("Metrics/acc", accuracy_score(all_labels, all_preds), epoch)
        writer.add_scalar("Metrics/recall", recall_score(all_labels, all_preds), epoch)
        writer.add_scalar("Metrics/precision", precision_score(all_labels, all_preds), epoch)
        writer.add_scalar("Metrics/f1", f1_score(all_labels, all_preds), epoch)
        writer.add_scalar("Loss/val", total_loss, epoch)

        if total_loss < best_loss:
            torch.save(model, save_path)
            best_loss = total_loss

def test(model, test_dataloader, writer):
    model.eval()
    all_preds = np.array([])
    all_labels = np.array([])

    for input_ids, attention_mask, labels in test_dataloader:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        all_preds = np.append(all_preds, out.logits.argmax(1).cpu().numpy())
        all_labels = np.append(all_labels, labels.cpu())

    writer.add_scalar("Test/acc", accuracy_score(all_labels, all_preds))
    writer.add_scalar("Test/recall", recall_score(all_labels, all_preds))
    writer.add_scalar("Test/precision", precision_score(all_labels, all_preds))
    writer.add_scalar("Test/f1", f1_score(all_labels, all_preds))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data preprocessing
    dga_dataset = pd.read_csv("../../data/dga_data.csv")
    dga_dataset["isDGA_flag"] = dga_dataset["isDGA"].apply(lambda x: int(x == "dga"))

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    encodings = tokenizer(list(dga_dataset["domain"].astype(str)), padding="max_length", truncation=True, max_length=64)

    X = pd.Series(zip(encodings["input_ids"], encodings["attention_mask"]))
    y = dga_dataset["isDGA_flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = list(zip(*X_train))
    X_val = list(zip(*X_val))
    X_test = list(zip(*X_test))

    train_dataloader = DataLoader(
        TensorDataset(torch.tensor(X_train[0], dtype=torch.long).to(device),
                      torch.tensor(X_train[1], dtype=torch.long).to(device),
                      torch.tensor(y_train.array, dtype=torch.long).to(device)),
        batch_size=128)
    val_dataloader = DataLoader(
        TensorDataset(torch.tensor(X_val[0], dtype=torch.long).to(device),
                      torch.tensor(X_val[1], dtype=torch.long).to(device),
                      torch.tensor(y_val.array, dtype=torch.long).to(device)),
        batch_size=128)
    test_dataloader = DataLoader(
        TensorDataset(torch.tensor(X_test[0], dtype=torch.long).to(device),
                      torch.tensor(X_test[1], dtype=torch.long).to(device),
                      torch.tensor(y_test.array, dtype=torch.long).to(device)),
        batch_size=128)

    model = CanineForSequenceClassification.from_pretrained("google/canine-c", num_labels=2)

    # for name, param in model.named_parameters():
    #     if not name.startswith("classifier"):
    #         param.requires_grad = False
    #     if param.requires_grad:
    #         print(name)

    model.to(device)
    epochs = 10
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter(log_dir="runs/finetune")
    save_path = "model.pth"

    train(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs, writer, save_path)
    model = torch.load(save_path, weights_only=False)
    test(model, test_dataloader, writer)

    writer.close()

if __name__ == "__main__":
    main()
