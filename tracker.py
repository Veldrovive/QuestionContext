from typing import List, Dict, Any
from pathlib import Path

import wandb
import torch


class WandBTracker:
    """
    
    """
    def __init__(self,
        project_name: str,
        run_name: str,
        config: dict = None,
        report_train_loss_every: int = 1,
        run_id: str = None
    ):
        if run_id:
            wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                id=run_id,
                resume="must"
            )
            print(f"Resuming run {run_id}")
        else:
            wandb.init(
                project=project_name,
                name=run_name,
                config=config,
            )
        self.config = config
        self.step = 0
        self.report_train_loss_every = report_train_loss_every
        self.train_loss_queue = []

    def increment_step(self):
        """
        During a step we actually upload the data to wandb that we have logged
        """
        self.step += 1

    def log_train_loss(self, loss: torch.tensor, epoch: int):
        """
        Records the loss for the train step
        """
        self.train_loss_queue.append(loss)
        if len(self.train_loss_queue) == self.report_train_loss_every:
            avg_train_loss = sum(self.train_loss_queue) / len(self.train_loss_queue)
            wandb.log({"train_loss": avg_train_loss, "epoch": epoch}, step=self.step)
            self.train_loss_queue = []

    def log_val_loss(self, loss: torch.tensor, epoch: int):
        """
        Records the loss for the val step
        """
        wandb.log({"val_loss": loss, "epoch": epoch}, step=self.step)


    def log_test_accuracy(self, top_1_score, top_5_score, top_10_score, epoch: int):
        """
        Records the loss for the test step
        """
        wandb.log({"top_1_score": top_1_score, "top_5_score": top_5_score, "top_10_score": top_10_score, "epoch": epoch}, step=self.step)

    def log_prediction_examples(self, examples: Dict[str, Dict[str, Any]], epoch: int):
        """
        Creates a table to log the model predictions

        Examples has the format: { question_id: { question: str, true_context: str, predicted_contexts: [str], true_prediction_index: int, out_of_unique_contexts: int } }

        All questions have the same number of predicted contexts
        """
        num_predictions = len(examples[list(examples.keys())[0]]["predicted_contexts"])
        table = wandb.Table(columns=["Question", "True Prediction Index", "Out Of", "True Context"] + [f"Prediction {i}" for i in range(num_predictions)])
        for example in examples.values():
            table.add_data(
                example["question"],
                example["true_prediction_index"],
                example["out_of_unique_contexts"],
                example["true_context"],
                *example["predicted_contexts"]
            )
        wandb.log({"prediction_examples": table, "epoch": epoch}, step=self.step)

    def save(self, path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        """
        Saves the model, optimizer, and scheduler
        """
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": self.step,
            "config": self.config
        }
        torch.save(save_dict, path)

    def load(self, path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """
        Loads the model, optimizer, and scheduler
        """
        save_dict = torch.load(path)
        model.load_state_dict(save_dict["model"])
        optimizer.load_state_dict(save_dict["optimizer"])
        self.step = save_dict["step"]
        return save_dict["epoch"]

    def clean_up(self):
        """
        Call this at the end of the run to close the wandb process
        """
        # If the train loss queue is not empty, log it
        if len(self.train_loss_queue) > 0:
            avg_train_loss = sum(self.train_loss_queue) / len(self.train_loss_queue)
            wandb.log({"train_loss": avg_train_loss}, step=self.step)
            self.train_loss_queue = []
        wandb.finish()
