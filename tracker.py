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
    ):
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
        )
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
