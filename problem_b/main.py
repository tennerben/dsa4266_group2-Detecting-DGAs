import string

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# Valid domain characters
VOCAB = string.ascii_lowercase + string.digits + "-."
stoi = {ch: i + 1 for i, ch in enumerate(VOCAB)}  # 0 = padding
PAD_IDX = 0
NUM_WORKERS = 4


class DGADataSet(Dataset):
    def __init__(self, domains, labels, seq_len):
        self.domains = domains.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.seq_len = seq_len

    def encode_domain(self, domain):
        domain = domain.lower()
        encoded = [stoi.get(c, 0) for c in domain]
        encoded = encoded[:self.seq_len]
        if len(encoded) < self.seq_len:
            encoded += [PAD_IDX] * (self.seq_len - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        domain = self.domains.iloc[idx]
        label = self.labels.iloc[idx]
        x = self.encode_domain(domain)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


class DGADataLoader(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
        )


class SampleModel(L.LightningModule):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128, kernel_size=5, lr=1e-3):
        super().__init__()
        self.lr = lr
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # CNN for character n-gram detection
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.fc = nn.Linear(num_filters, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        """x shape: (batch, seq_len)"""
        x = self.embedding(x)           # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)          # (batch, embed_dim, seq_len)
        x = F.relu(self.conv(x))        # (batch, filters, L)
        x = F.max_pool1d(x, x.size(2))  # global max pool
        x = x.squeeze(2)                # (batch, filters)
        logits = self.fc(x)             # (batch, 1)
        return logits.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y.bool()).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y.bool()).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y.bool()).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)


class Logger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics["train_loss"].item()
        print(f"Epoch {trainer.current_epoch}: Train Loss: {loss}")
        self.train_loss.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics["val_loss"].item()
        print(f"Epoch {trainer.current_epoch}: Validation Loss: {loss}")
        self.val_loss.append(loss)

    def on_test_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics["test_loss"].item()
        print(f"Epoch {trainer.current_epoch}: Test Loss: {loss}")
        self.test_loss.append(loss)


def main():
    # Data preprocessing
    dga_dataset = pd.read_csv("data/dga_data.csv")
    X = dga_dataset["domain"]  # Pure text
    dga_dataset["isDGA_flag"] = dga_dataset["isDGA"].apply(lambda x: x == "dga")
    y = dga_dataset["isDGA_flag"]   # Binary

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Build datamodule
    seq_len = 75
    vocab_size = len(stoi) + 1
    datamodule = DGADataLoader(
        DGADataSet(X_train, y_train, seq_len),
        DGADataSet(X_val, y_val, seq_len),
        DGADataSet(X_test, y_test, seq_len),
    )

    # Train
    logger = Logger()
    model = SampleModel(vocab_size=vocab_size)
    trainer = Trainer(
        max_epochs=10,
        fast_dev_run=True,
        accelerator="auto",
        callbacks=[logger],
        gradient_clip_val=1.0,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
