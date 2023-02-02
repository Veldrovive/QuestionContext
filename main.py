from torch.utils.data import random_split
import torch
import argparse

from dataloader import SquadLocalContextContrastiveDataset, get_sets
from model import QClip
from trainer import Trainer
from tracker import WandBTracker

import open_clip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-out-of", type=int, default=100)
    parser.add_argument("--clip-arch", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--clip-checkpoint", type=str, default="laion400m_e32")
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--projection-layers", type=int, default=1)

    parser.add_argument("--answer-projection", action='store_true')
    parser.add_argument('--no-answer-projection', dest='answer-projection', action='store_false')

    parser.add_argument("--question-projection", action='store_true')
    parser.add_argument('--no-question-projection', dest='question-projection', action='store_false')

    parser.add_argument("--simple-loss", action='store_true')
    parser.add_argument('--no-simple-loss', dest='simple-loss', action='store_false')

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    # Load the datasets and CLIP model for preprocessing
    train_set, test_set = get_sets()
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_arch, pretrained=args.clip_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.clip_arch)
    embedding_width = model.encode_text(tokenizer(["hello", "world"]).to(device)).shape[1]
    print(f"Embedding width: {embedding_width}")

    train_val_dataset = SquadLocalContextContrastiveDataset(train_set, model, tokenizer, normalize_clip=True, clip_device=device)
    train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
    test_dataset = SquadLocalContextContrastiveDataset(test_set, model, tokenizer, normalize_clip=True, clip_device=device)

    # Configure and create the question-answer clip model
    model_config = {
        "embedding_dim": embedding_width,
        "output_dim": args.output_dim,
        "projection_layers": args.projection_layers,
        "use_answer_projection": args.answer_projection,
        "use_question_projection": args.question_projection,
        "simple_loss": args.simple_loss,
        "temperature": args.temperature
    }
    qclip = QClip(**model_config)
    qclip.to(device)

    # Configure the wandb tracker and pass context about this run
    if args.run_name:
        run_name = args.run_name
    else:
        print(f"Enter run name for wandb:")
        run_name = input()
    tracker = WandBTracker("QuestionContext", run_name, {
        **model_config,
        "dataset": "SQuAD",
        "clip_arch": args.clip_arch,
        "clip_checkpoint": args.clip_checkpoint,
        "batch_size": args.batch_size,
        "test_out_of": args.test_out_of,
        "lr": args.lr
    }, report_train_loss_every=1)

    trainer = Trainer(
        train_dataset,
        val_dataset,
        test_dataset,
        qclip,
        tracker = tracker,
        batch_size = args.batch_size,
        shuffle = True,
        save_dir = args.save_dir,
        lr = args.lr
    )

    trainer.train(
        epochs = 50,
        train_sample_limit = None,
        val_sample_limit = None,
        test_out_of=args.test_out_of,
        test_sample_limit = None
    )
