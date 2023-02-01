from torch import nn
import torch.nn.functional as F
import torch

class QProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class AProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class QClip(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        use_question_projection: bool = True,
        use_answer_projection: bool = True,
        temperature: float = 1.0,
        simple_loss: bool = True,
        return_loss: bool = False,
    ):
        """
        Creates the projection models for the question and answer embeddings if requested. Otherwise add an identity layer.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.simple_loss = simple_loss
        self.return_loss = return_loss
        self.q_projection = QProjection(embedding_dim, output_dim) if use_question_projection else nn.Identity()
        self.a_projection = AProjection(embedding_dim, output_dim) if use_answer_projection else nn.Identity()

    def __softmax_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str="none") -> torch.Tensor:
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(logits)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, batch):
        """
        Assume batch is a tuple of question and answer embeddings
        Dimensions of both should be (*, embedding_dim)
        """
        logits = (batch[0] @ batch[1].T) / self.temperature

        if self.return_loss:
            if self.simple_loss:
                loss = nn.CrossEntropyLoss()(logits, torch.arange(logits.shape[0]))
            else:
                q_self_similarity = (batch[0] @ batch[0].T)
                a_self_similarity = (batch[1] @ batch[1].T)
                target = F.softmax(
                    (q_self_similarity + a_self_similarity) / 2 * self.temperature, dim=-1
                )
                q_loss = self.__softmax_cross_entropy(logits, target, reduction="none")
                a_loss = self.__softmax_cross_entropy(logits.T, target.T, reduction="none")
                loss = (q_loss + a_loss) / 2
            return logits, loss.mean()
        else:
            return logits




