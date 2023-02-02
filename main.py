from torch.utils.data import random_split
import torch

from dataloader import SquadLocalContextContrastiveDataset, get_sets
from model import QClip
from trainer import Trainer
from tracker import WandBTracker

import open_clip

if __name__ == "__main__":
    batch_size = 1024
    test_out_of = 500

    # Load the datasets and CLIP model for preprocessing
    train_set, test_set = get_sets()
    clip_arch = 'ViT-B-32-quickgelu'
    clip_checkpoint = 'laion400m_e32'
    model, _, preprocess = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = open_clip.get_tokenizer(clip_arch)
    embedding_width = model.encode_text(tokenizer(["hello", "world"]).to(device)).shape[1]
    print(f"Embedding width: {embedding_width}")

    train_val_dataset = SquadLocalContextContrastiveDataset(train_set, model, tokenizer, normalize_clip=True, clip_device=device)
    train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
    test_dataset = SquadLocalContextContrastiveDataset(test_set, model, tokenizer, normalize_clip=True, clip_device=device)

    # Configure and create the question-answer clip model
    model_config = {
        "embedding_dim": embedding_width,
        "output_dim": 512,
        "use_answer_projection": True,
        "use_question_projection": True,
        "simple_loss": True,
        "temperature": 1.0
    }
    qclip = QClip(**model_config)
    qclip.to(device)

    # Configure the wandb tracker and pass context about this run
    print(f"Enter run name for wandb:")
    run_name = input()
    tracker = WandBTracker("QuestionContext", run_name, {
        **model_config,
        "dataset": "SQuAD",
        "clip_arch": clip_arch,
        "clip_checkpoint": clip_checkpoint,
        "batch_size": batch_size,
        "test_out_of": test_out_of
    }, report_train_loss_every=1)

    trainer = Trainer(
        train_dataset,
        val_dataset,
        test_dataset,
        qclip,
        tracker = tracker,
        batch_size = batch_size,
        shuffle = True,
        save_dir = "./checkpoints"
    )

    trainer.train(
        epochs = 50,
        train_sample_limit = None,
        val_sample_limit = None,
        test_out_of=test_out_of,
        test_sample_limit = None
    )
