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
        save_dir: str = None,
        lr: float = 1e-3
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.model.return_loss = True
        self.save_dir = save_dir

        if not self.model.identity:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.batch_size = batch_size
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        self.tracker = tracker

    def _to_model_device(self, sample):
        if self.model.identity:
            return sample
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
            _, loss = self.model(sample[:2])
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
            _, loss = self.model(sample[:2])
            total_loss += loss.item()
            total_steps += 1
        print(f"Finished eval: {epoch}\n------------\n")
        if self.tracker is not None:
            self.tracker.log_val_loss(total_loss / total_steps, epoch)
        return total_loss / total_steps

    def _get_unique_answer_subset(self, sample):
            """
            Samples may have duplicate answers because the same sentence may answer mutliple questions. This is a problem for evaluation because the model may get the correct answer and we mark it wrong.
            To fix this, we remove the duplicate answers and instead map multiple questions to the same answer index.

            Returns sample with sample[1] shortened and a new sample[5] that is a ndarray of the new answer indices. This serves as a mapping from the old answer indices to the new answer indices.
            """
            answer_embeddings = sample[1]
            answer_text = sample[3]

            first_answer_map = {}
            unique_answer_indices = []
            first_answer_list = []
            for idx, answer in enumerate(answer_text):
                if answer not in first_answer_map:
                    unique_answer_indices.append(idx)
                    first_answer_map[answer] = len(unique_answer_indices) - 1
                    first_answer_list.append(len(unique_answer_indices) - 1)
                else:
                    first_answer_list.append(first_answer_map[answer])
            unique_answer_indices = torch.tensor(unique_answer_indices, dtype=torch.long, device=answer_embeddings.device)
            first_answer_list = np.array(first_answer_list, dtype=np.int32)
            unique_answer_text = []
            for idx in unique_answer_indices:
                unique_answer_text.append(answer_text[idx])

            unique_answer_embeddings = answer_embeddings[unique_answer_indices]
            # print(answer_text)
            # print(unique_answer_indices)
            # print(first_answer_list)
            # print(f"Unique answers: {len(unique_answer_embeddings)} out of {len(answer_embeddings)}")
            return [sample[0], unique_answer_embeddings, sample[2], unique_answer_text, sample[4], first_answer_list]

    def run_test(self, epoch: int, out_of: int = 100, sample_limit: int = None, example_prediction_limit: int = 10, dataset: SquadLocalContextContrastiveDataset = None):
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
        predictions = {}  # { question_id: { question: str, true_context: str, predicted_contexts: [str], true_prediction_index: int } }
        for sample in progress_bar:
            if sample_limit is not None and total_samples >= sample_limit:
                break
            total_samples += len(sample[0])
            sample = self._get_unique_answer_subset(sample)  # Removes duplicate answers from the answer embeddings. sample[5] is a ndarray of the new answer indices
            with torch.no_grad():
                sample = self._to_model_device(sample)
                question_projection = self.model.q_projection(sample[0])
                answer_projection = self.model.a_projection(sample[1])
                similarities = question_projection @ answer_projection.T

            similarities = similarities.detach().cpu().numpy()
            similarity_indices = similarities.argsort(axis=1)
            similarity_indices = similarity_indices[:, ::-1]
            # Now we can get the index of the true answer
            # To do this we subtract the row index from the similarity indices. The position of the true answer will be 0
            # zerod_similarity_indices = similarity_indices - np.arange(similarity_indices.shape[0]).reshape(-1, 1)
            zerod_similarity_indices = similarity_indices - sample[5].reshape(-1, 1)
            correct_answer_positions = np.where(zerod_similarity_indices == 0)[1]  # 0 Means the true answer is the top prediction, 4 means it's the 5th prediction, and so on
            top_1_count += np.sum(correct_answer_positions == 0)
            top_5_count += np.sum(correct_answer_positions < 5)
            top_10_count += np.sum(correct_answer_positions < 10)


            questions, contexts, qids, unique_context_map = sample[2], sample[3], sample[4], sample[5]
            for i in range(len(questions)):
                qid = qids[i]
                question = questions[i]
                true_context = contexts[unique_context_map[i]]
                predicted_contexts = [contexts[j] for j in similarity_indices[i][:example_prediction_limit]]
                true_prediction_index = correct_answer_positions[i]
                predictions[qid] = {
                    "question": question,
                    "true_context": true_context,
                    "predicted_contexts": predicted_contexts,
                    "true_prediction_index": true_prediction_index,
                    "out_of_unique_contexts": len(contexts)
                }

            progress_bar.set_description(f"Top 1: {top_1_count / total_samples:.4f} Top 5: {top_5_count / total_samples:.4f} Top 10: {top_10_count / total_samples:.4f}")

        if self.tracker is not None:
            self.tracker.log_test_accuracy(top_1_count / total_samples, top_5_count / total_samples, top_10_count / total_samples, epoch)
            self.tracker.log_prediction_examples(predictions, epoch)
        print(f"Finished test: {epoch}\n------------\n")
        return top_1_count / total_samples, top_5_count / total_samples, top_10_count / total_samples

    def save(self, epoch: int):
        if self.save_dir is None:
            return
        save_path = Path(self.save_dir) / f"epoch_{epoch}.pt"
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        self.tracker.save(save_path, model=self.model, optimizer=self.optimizer, epoch=epoch)

    def train(self,
        epochs: int,
        start_epoch: int = 0,
        train_sample_limit: int = None,
        val_sample_limit: int = None,
        test_out_of: int = 100,
        test_sample_limit: int = None,
    ):
        if self.model.identity:
            top_1, top_5, top_10 = self.run_test(0, out_of=test_out_of, sample_limit=test_sample_limit)
        else:
            for epoch in range(start_epoch, epochs):
                train_loss = self.run_epoch(epoch, train_sample_limit)
                print(f"Average Train loss: {train_loss}")
                val_loss = self.run_eval(epoch, val_sample_limit)
                print(f"Average Val loss: {val_loss}")
                top_1, top_5, top_10 = self.run_test(epoch, out_of=test_out_of, sample_limit=test_sample_limit)
                print(f"Top 1: {top_1} Top 5: {top_5} Top 10: {top_10}")
                self.save(epoch+1)

        if self.tracker is not None:
            self.tracker.clean_up()

            

