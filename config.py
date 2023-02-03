from pydantic import BaseModel, Field
from typing import Optional, List
import git
import os

from enum import Enum
class Dataset(str, Enum):
    SQUAD = "squad"
    QUAC = "quac"

def get_name():
    print("Enter run name for wandb:")
    return input()

def get_git_commit():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(file_dir, search_parent_directories=True)
    return repo.head.object.hexsha

def get_slurm_job_id():
    import os
    return os.getenv("SLURM_JOB_ID")

class RunConfig(BaseModel):
    run_name: str = Field(default_factory=get_name, description="Name of the run")
    resume_id: Optional[str] = Field(None, description="Resume run ID")
    save_dir: str = Field("./checkpoints", description="Directory to save checkpoints")
    save_latest: bool = Field(True, description="Save the latest checkpoint")
    save_all: bool = Field(False, description="Save all checkpoints")

    epochs: int = Field(30, description="Number of epochs to train for")
    train_sample_limit: int = Field(None, description="Limit the number of training samples")
    val_sample_limit: int = Field(None, description="Limit the number of validation samples")
    test_sample_limit: int = Field(None, description="Limit the number of test samples")
    test_out_of: int = Field(100, description="Now many examples to present to the model to select from")

class DataConfig(BaseModel):
    context_radius: int = Field(0, description="Number of sentences around the answer to include in the context")
    clip_arch: str = Field("ViT-B-32", description="Architecture of the clip model")
    clip_checkpoint: str = Field("laion400m_e32", description="Checkpoint of the clip model")
    train_split: float = Field(0.8, description="Fraction of the data to use for training")
    datasets: List[Dataset] = Field([Dataset.SQUAD], description="Datasets to use")

class ModelConfig(BaseModel):
    output_dim: int = Field(512, description="Output dimension of the projection model")
    hidden_dim: int = Field(768, description="Hidden dimension of the projection model")
    num_layers: int = Field(2, description="Number of layers in the projection model")

    context_projection: bool = Field(False, description="Whether to project the context embedding")
    question_projection: bool = Field(True, description="Whether to project the question embedding")

class TrainConfig(BaseModel):
    batch_size: int = Field(64, description="Batch size")
    simple_loss: bool = Field(True, description="Whether to use a simple loss function")
    temperature: float = Field(1.0, description="Temperature for the loss function")
    lr: float = Field(1e-3, description="Learning rate")

class RuntimeConfig(BaseModel):
    """
    Gets values from environment variables and other sources automatically
    """
    git_commit: str = Field(default_factory=get_git_commit, description="Git commit hash")
    slurm_job_id: str = Field(default_factory=get_slurm_job_id, description="SLURM job ID")
    wandb_run_id: str = Field(None, description="Wandb run ID")

class Config(BaseModel):
    run: RunConfig = Field(description="Run configuration", required=True)
    data: DataConfig = Field(description="Data configuration", required=True)
    model: ModelConfig = Field(description="Model configuration", required=True)
    train: TrainConfig = Field(description="Training configuration", required=True)
    runtime: RuntimeConfig = Field(description="Runtime configuration", default_factory=RuntimeConfig)


if __name__ == "__main__":
    import json
    json_input = json.load(open("/fsx/aidan/new/qa/QuestionContext/config/example.json"))
    config = Config(**json_input)
    print(config.json(indent=4))
                                   
