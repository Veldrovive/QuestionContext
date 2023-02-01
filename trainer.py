from torch.utils.data import DataLoader
import torch

from dataloader import SquadLocalContextContrastiveDataset
from model import QClip

class Trainer:
    def __init__(
        self,
        train_dataset: SquadLocalContextContrastiveDataset,
        val_dataset: SquadLocalContextContrastiveDataset,
        model: QClip,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.model.return_loss = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.batch_size = batch_size
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    def run_epoch(self, epoch: int, sample_limit: int = None):
        print(f"\n------------\nStarting epoch: {epoch}")
        self.model.train()
        total_loss = 0
        total_samples = 0
        for sample_idx, sample in enumerate(self.train_dataloader):
            if sample_limit is not None and sample_idx >= sample_limit:
                break
            print(f"Sample {sample_idx}")
            print(sample[0].shape, sample[1].shape)
            self.optimizer.zero_grad()
            logits, loss = self.model(sample)
            print(f"Loss: {loss}")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_samples += sample[0].shape[0]
        print(f"Finished epoch: {epoch}\n------------\n")
        return total_loss / total_samples

    def run_eval(self, epoch: int):
        print(f"\n------------\nStarting eval: {epoch}")
        self.model.eval()
        total_loss = 0
        total_samples = 0
        for sample_idx, sample in enumerate(self.val_dataloader):
            print(f"Sample {sample_idx}")
            print(sample[0].shape, sample[1].shape)
            logits, loss = self.model(sample)
            print(f"Loss: {loss}")
            total_loss += loss.item()
            total_samples += sample[0].shape[0]
        print(f"Finished eval: {epoch}\n------------\n")
        return total_loss / total_samples

    def train(self, epochs: int):
        print("Starting training")
        for epoch in range(epochs):
            train_loss = self.run_epoch(epoch)
            print(f"Average Train loss: {train_loss}")
            val_loss = self.run_eval(epoch)
            print(f"Average Val loss: {val_loss}")

            

