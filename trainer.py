from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path

from dataloader import SquadLocalContextContrastiveDataset
from model import QClip
from tracker import WandBTracker

class Trainer:
    def __init__(
        self,
        train_dataset: SquadLocalContextContrastiveDataset,
        val_dataset: SquadLocalContextContrastiveDataset,
        test_dataset: SquadLocalContextContrastiveDataset,
        model: QClip,
        tracker: WandBTracker = None,
        batch_size: int = 32,
        shuffle: bool = True,
        save_dir: str = None
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.model.return_loss = True
        self.save_dir = save_dir

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.batch_size = batch_size
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        self.tracker = tracker

    def _to_model_device(self, sample):
        model_device = next(self.model.parameters()).device
        sample[0] = sample[0].to(model_device)
        sample[1] = sample[1].to(model_device)
        return sample

    def run_epoch(self, epoch: int, sample_limit: int = None):
        print(f"\n------------\nStarting epoch: {epoch}")
        self.model.train()
        total_loss = 0
        total_steps = 0
        predicted_name_samples = len(self.train_dataset) if sample_limit is None else min(sample_limit, len(self.train_dataset))
        progress_bar = tqdm(enumerate(self.train_dataloader), total=int(round(predicted_name_samples / self.batch_size)))
        for sample_idx, sample in progress_bar:
            if sample_limit is not None and sample_idx * self.batch_size >= sample_limit:
                break
            sample = self._to_model_device(sample)
            self.optimizer.zero_grad()
            _, loss = self.model(sample)
            if self.tracker is not None:
                self.tracker.log_train_loss(loss.item(), epoch)
                self.tracker.increment_step()
            progress_bar.set_description(f"Loss: {round(loss.item(), 5)}")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_steps += 1
        print(f"Finished epoch: {epoch}\n------------\n")
        return total_loss / total_steps

    def run_eval(self, epoch: int, sample_limit: int = None):
        print(f"\n------------\nStarting eval: {epoch}")
        self.model.eval()
        total_loss = 0
        total_steps = 0
        predicted_name_samples = len(self.val_dataset) if sample_limit is None else min(sample_limit, len(self.val_dataset))
        progress_bar = tqdm(enumerate(self.val_dataloader), total=int(round(predicted_name_samples / self.batch_size)))
        for sample_idx, sample in progress_bar:
            if sample_limit is not None and sample_idx * self.batch_size >= sample_limit:
                break
            sample = self._to_model_device(sample)
            _, loss = self.model(sample)
            total_loss += loss.item()
            total_steps += 1
        print(f"Finished eval: {epoch}\n------------\n")
        if self.tracker is not None:
            self.tracker.log_val_loss(total_loss / total_steps, epoch)
        return total_loss / total_steps

    def run_test(self, epoch: int, out_of: int = 100, sample_limit: int = None, dataset: SquadLocalContextContrastiveDataset = None):
        """
        To test the model, we use a different strategy. We want to get the top 1, top 5, and top 10 score for correctly finding the context sentence given the question
        To do this, we generate n samples from the dataloader and run the model on them. We then do an argsort on the similarities to get the top 10 in each row.
        The true answer will be on the diagonal, so we can then calculate the top 1, top 5, and top 10 scores.
        """
        if dataset is None:
            dataset = self.test_dataset
        dataloader = DataLoader(dataset, batch_size=out_of, shuffle=False)
        self.model.eval()
        print(f"\n------------\nStarting test: {epoch}")
        total_samples = 0
        top_1_count = 0
        top_5_count = 0
        top_10_count = 0
        predicted_name_samples = len(dataset) if sample_limit is None else min(sample_limit, len(dataset))
        progress_bar = tqdm(dataloader, total=int(round(predicted_name_samples / out_of)))
        for sample in progress_bar:
            if sample_limit is not None and total_samples >= sample_limit:
                break
            total_samples += len(sample[0])
            with torch.no_grad():
                sample = self._to_model_device(sample)
                similarities, _ = self.model(sample)

            similarities = similarities.detach().cpu().numpy()
            similarity_indices = similarities.argsort(axis=1)
            similarity_indices = similarity_indices[:, ::-1]
            # Now we can get the index of the true answer
            # To do this we subtract the row index from the similarity indices. The position of the true answer will be 0
            zerod_similarity_indices = similarity_indices - np.arange(similarity_indices.shape[0]).reshape(-1, 1)
            correct_answer_positions = np.where(zerod_similarity_indices == 0)[1]
            top_1_count += np.sum(correct_answer_positions == 0)
            top_5_count += np.sum(correct_answer_positions < 5)
            top_10_count += np.sum(correct_answer_positions < 10)

            progress_bar.set_description(f"Top 1: {top_1_count / total_samples:.4f} Top 5: {top_5_count / total_samples:.4f} Top 10: {top_10_count / total_samples:.4f}")

        if self.tracker is not None:
            self.tracker.log_test_accuracy(top_1_count / total_samples, top_5_count / total_samples, top_10_count / total_samples, epoch)
        print(f"Finished test: {epoch}\n------------\n")
        return top_1_count / total_samples, top_5_count / total_samples, top_10_count / total_samples

    def save(self, epoch: int):
        if self.save_dir is None:
            return
        save_path = Path(self.save_dir) / f"epoch_{epoch}.pt"
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        self.model.save(save_path)

    def train(self,
        epochs: int,
        train_sample_limit: int = None,
        val_sample_limit: int = None,
        test_out_of: int = 100,
        test_sample_limit: int = None,
    ):
        for epoch in range(epochs):
            train_loss = self.run_epoch(epoch, train_sample_limit)
            print(f"Average Train loss: {train_loss}")
            val_loss = self.run_eval(epoch, val_sample_limit)
            print(f"Average Val loss: {val_loss}")
            top_1, top_5, top_10 = self.run_test(epoch, out_of=test_out_of, sample_limit=test_sample_limit)
            print(f"Top 1: {top_1} Top 5: {top_5} Top 10: {top_10}")
            self.save(epoch)

        if self.tracker is not None:
            self.tracker.clean_up()

            

