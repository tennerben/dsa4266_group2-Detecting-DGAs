import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformer import train
from transformers import CanineTokenizer, CanineForSequenceClassification


def get_loaders(family, device, X_train, y_train, X_val, y_val):
    X_train_ids = []
    X_train_atn = []
    y_train_ids = []
    X_test_ids = []
    X_test_atn = []
    y_test_ids = []
    for i, row in enumerate(X_train):
        if row[2] == family:
            X_test_ids.append(row[0])
            X_test_atn.append(row[1])
            y_test_ids.append(y_train[i])
        else:
            X_train_ids.append(row[0])
            X_train_atn.append(row[1])
            y_train_ids.append(y_train[i])

    train_dataloader = DataLoader(
        TensorDataset(torch.tensor(X_train_ids, dtype=torch.long).to(device),
                      torch.tensor(X_train_atn, dtype=torch.long).to(device),
                      torch.tensor(y_train_ids, dtype=torch.long).to(device)),
        batch_size=128
    )

    X_val_ids = []
    X_val_atn = []
    y_val_ids = []
    for i, row in enumerate(X_val):
        if row[2] == family:
            X_test_ids.append(row[0])
            X_test_atn.append(row[1])
            y_test_ids.append(y_val[i])
        else:
            X_val_ids.append(row[0])
            X_val_atn.append(row[1])
            y_val_ids.append(y_val[i])
    val_dataloader = DataLoader(
        TensorDataset(torch.tensor(X_val_ids, dtype=torch.long).to(device),
                      torch.tensor(X_val_atn, dtype=torch.long).to(device),
                      torch.tensor(y_val_ids, dtype=torch.long).to(device)),
        batch_size=128
    )

    test_dataloader = DataLoader(
        TensorDataset(torch.tensor(X_test_ids, dtype=torch.long).to(device),
                      torch.tensor(X_test_atn, dtype=torch.long).to(device),
                      torch.tensor(y_test_ids, dtype=torch.long).to(device)),
        batch_size=128
    )

    return train_dataloader, val_dataloader, test_dataloader


def test(model, test_dataloader, family, writer):
    model.eval()
    all_preds = np.array([])
    all_labels = np.array([])

    for input_ids, attention_mask, labels in test_dataloader:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        all_preds = np.append(all_preds, out.logits.argmax(1).cpu().numpy())
        all_labels = np.append(all_labels, labels.cpu())

    writer.add_scalar(f"Test/{family}_acc", accuracy_score(all_labels, all_preds))
    writer.add_scalar(f"Test/{family}_recall", recall_score(all_labels, all_preds))
    writer.add_scalar(f"Test/{family}_precision", precision_score(all_labels, all_preds))
    writer.add_scalar(f"Test/{family}_f1", f1_score(all_labels, all_preds))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data preprocessing
    dga_dataset = pd.read_csv("../../data/dga_data.csv")
    dga_dataset["isDGA_flag"] = dga_dataset["isDGA"].apply(lambda x: int(x == "dga"))

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    encodings = tokenizer(list(dga_dataset["domain"].astype(str)), padding="max_length", truncation=True, max_length=64)

    X = list(zip(encodings["input_ids"], encodings["attention_mask"], dga_dataset["subclass"]))
    y = dga_dataset["isDGA_flag"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    dga_families = dga_dataset["subclass"][dga_dataset["isDGA_flag"] == 1].unique()
    print(dga_families)

    for family in dga_families:
        train_dataloader, val_dataloader, test_dataloader = get_loaders(family, device, X_train, y_train, X_val, y_val)

        model = CanineForSequenceClassification.from_pretrained("google/canine-c", num_labels=2)
        model.to(device)
        epochs = 10
        optimizer = AdamW(model.parameters(), lr=5e-5)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        writer = SummaryWriter(log_dir=f"runs/{family}")
        save_path = f"{family}.pth"

        train(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs, writer, save_path)
        model = torch.load(save_path, weights_only=False)
        test(model, test_dataloader, family, writer)
        writer.close()


if __name__ == "__main__":
    main()
