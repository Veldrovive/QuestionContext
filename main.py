from torch.utils.data import random_split
import torch
import argparse
import json
from pathlib import Path

from dataloader import SquadLocalContextContrastiveDataset, QuacLocalContextContrastiveDataset, get_quac_sets, get_squad_sets
from model import QClip
from trainer import Trainer
from tracker import WandBTracker
from config import Config, Dataset

import open_clip

def generate_model(config: Config, device: torch.device = torch.device("cpu")):
    """
    Generates the qclip and clip models
    """
    clip_model, _, _ = open_clip.create_model_and_transforms(config.data.clip_arch, pretrained=config.data.clip_checkpoint)
    clip_model.to(device)

    tokenizer = open_clip.get_tokenizer(config.data.clip_arch)
    embedding_width = clip_model.encode_text(tokenizer(["hello", "world"]).to(device)).shape[1]
    print(f"Embedding width: {embedding_width}")

    qclip = QClip(
        embedding_width,
        config.model.output_dim,
        config.model.hidden_dim,
        projection_layers = config.model.num_layers,
        use_question_projection = config.model.question_projection,
        use_answer_projection = config.model.context_projection,
        temperature = config.train.temperature,
        simple_loss = config.train.simple_loss,
        return_loss = True,
    )
    qclip.to(device)

    return qclip, clip_model

def load_checkpoint(checkpoint: Path, config: Config = None, device: torch.device = torch.device("cpu")):
    """
    Loads a checkpoint from the given path and returns the model, config, and start epoch
    """
    if config is not None:
        print("Overriding checkpoint config with provided config")
    checkpoint = torch.load(checkpoint, map_location=device)
    model_state_dict = checkpoint["model"]
    optimizer_state_dict = checkpoint["optimizer"]
    start_epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    if config is None:
        config = Config(**checkpoint["config"])

    qclip, clip_model = generate_model(config, device)
    qclip.load_state_dict(model_state_dict)

    return qclip, clip_model, config, start_epoch, step, optimizer_state_dict
    
def get_datasets(config: Config, clip_model):
    """
    
    """
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for dataset in config.data.datasets:
        if dataset == Dataset.SQUAD:
            train_set, test_set = get_squad_sets()
            clip_device = next(clip_model.parameters()).device
            tokenizer = open_clip.get_tokenizer(config.data.clip_arch)
            train_val_dataset = SquadLocalContextContrastiveDataset(train_set, config.data.context_radius, clip_model, tokenizer, normalize_clip=True, clip_device=clip_device)
            train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
            test_dataset = SquadLocalContextContrastiveDataset(test_set, config.data.context_radius, clip_model, tokenizer, normalize_clip=True, clip_device=clip_device)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        elif dataset == Dataset.QUAC:
            train_set, test_set = get_quac_sets()
            clip_device = next(clip_model.parameters()).device
            tokenizer = open_clip.get_tokenizer(config.data.clip_arch)
            train_val_dataset = QuacLocalContextContrastiveDataset(train_set, config.data.context_radius, clip_model, tokenizer, normalize_clip=True, clip_device=clip_device)
            train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
            test_dataset = QuacLocalContextContrastiveDataset(test_set, config.data.context_radius, clip_model, tokenizer, normalize_clip=True, clip_device=clip_device)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        else:
            raise ValueError(f"Unknown dataset {dataset}")
        
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    return train_dataset, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = None
    if args.config is not None:
        config = Config(**json.load(open(args.config)))

    qclip = None
    start_epoch = 0
    if args.checkpoint is not None:
        qclip, clip_model, config, start_epoch, step, optimizer_state_dict = load_checkpoint(args.checkpoint, config, device)
    else:
        qclip, clip_model = generate_model(config, device)

    train, val, test = get_datasets(config, clip_model)
    
    tracker = WandBTracker(
        "QuestionContext", 
        config.run.run_name,
        report_train_loss_every=1,
        run_id=config.run.resume_id,
        config=config,
    )

    trainer = Trainer(
        train,
        val,
        test,
        qclip,
        config,
        tracker = tracker,
        shuffle = True
    )

    if args.checkpoint is not None:
        trainer.load_optimizer_state_dict(optimizer_state_dict)
        tracker.step = step

    print("Training with config:")
    print(config.json(indent=4))
    
    trainer.train(
        epochs = config.run.epochs,
        start_epoch = start_epoch,
        train_sample_limit = config.run.train_sample_limit,
        val_sample_limit = config.run.val_sample_limit,
        test_out_of = config.run.test_out_of,
        test_sample_limit = config.run.test_sample_limit,
    )

if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--run-name", type=str, default=None)
    # parser.add_argument("--save-dir", type=str, default=None)
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--batch-size", type=int, default=64)
    # parser.add_argument("--train-sample-limit", type=int, default=None)
    # parser.add_argument("--val-sample-limit", type=int, default=None)
    # parser.add_argument("--test-sample-limit", type=int, default=None)
    # parser.add_argument("--test-out-of", type=int, default=100)

    # parser.add_argument("--context-radius", type=int, default=1)
    # parser.add_argument("--clip-arch", type=str, default="ViT-B-32-quickgelu")
    # parser.add_argument("--clip-checkpoint", type=str, default="laion400m_e32")
    # parser.add_argument("--output-dim", type=int, default=512)
    # parser.add_argument("--projection-layers", type=int, default=1)

    # parser.add_argument("--answer-projection", action='store_true')
    # parser.add_argument('--no-answer-projection', dest='answer-projection', action='store_false')

    # parser.add_argument("--question-projection", action='store_true')
    # parser.add_argument('--no-question-projection', dest='question-projection', action='store_false')

    # parser.add_argument("--simple-loss", action='store_true')
    # parser.add_argument('--no-simple-loss', dest='simple-loss', action='store_false')

    # parser.add_argument("--temperature", type=float, default=1.0)
    # parser.add_argument("--lr", type=float, default=1e-3)

    # parser.add_argument("--checkpoint", type=str, default=None)
    # parser.add_argument("--resume-run-id", type=str, default=None)

    # args = parser.parse_args()

    # # Load the datasets and CLIP model for preprocessing
    # train_set, test_set = get_sets()
    # model, _, preprocess = open_clip.create_model_and_transforms(args.clip_arch, pretrained=args.clip_checkpoint)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # tokenizer = open_clip.get_tokenizer(args.clip_arch)
    # embedding_width = model.encode_text(tokenizer(["hello", "world"]).to(device)).shape[1]
    # print(f"Embedding width: {embedding_width}")

    # train_val_dataset = SquadLocalContextContrastiveDataset(train_set, args.context_radius, model, tokenizer, normalize_clip=True, clip_device=device)
    # train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
    # test_dataset = SquadLocalContextContrastiveDataset(test_set, args.context_radius, model, tokenizer, normalize_clip=True, clip_device=device)

    # # Configure and create the question-answer clip model
    # model_config = {
    #     "embedding_dim": embedding_width,
    #     "output_dim": args.output_dim,
    #     "projection_layers": args.projection_layers,
    #     "use_answer_projection": args.answer_projection,
    #     "use_question_projection": args.question_projection,
    #     "simple_loss": args.simple_loss,
    #     "temperature": args.temperature
    # }
    # qclip = QClip(**model_config)
    # qclip.to(device)

    # # Configure the wandb tracker and pass context about this run
    # if args.run_name:
    #     run_name = args.run_name
    # else:
    #     print(f"Enter run name for wandb:")
    #     run_name = input()
    # tracker = WandBTracker("QuestionContext", run_name, {
    #     **model_config,
    #     "dataset": "SQuAD",
    #     "clip_arch": args.clip_arch,
    #     "clip_checkpoint": args.clip_checkpoint,
    #     "batch_size": args.batch_size,
    #     "test_out_of": args.test_out_of,
    #     "lr": args.lr,
    #     "context_radius": args.context_radius,
    # }, report_train_loss_every=1, run_id=args.resume_run_id)

    # trainer = Trainer(
    #     train_dataset,
    #     val_dataset,
    #     test_dataset,
    #     qclip,
    #     tracker = tracker,
    #     batch_size = args.batch_size,
    #     shuffle = True,
    #     save_dir = args.save_dir,
    #     lr = args.lr
    # )

    # start_epoch = 0
    # if args.checkpoint:
    #     start_epoch = tracker.load(args.checkpoint, qclip, trainer.optimizer)
    #     print(f"Loaded checkpoint from {args.checkpoint} at epoch {start_epoch} and step {tracker.step}.")

    # trainer.train(
    #     epochs = 50,
    #     start_epoch = start_epoch,
    #     train_sample_limit = args.train_sample_limit,
    #     val_sample_limit = args.val_sample_limit,
    #     test_out_of=args.test_out_of,
    #     test_sample_limit = args.test_sample_limit
    # )
