from dataloader import SquadLocalContextContrastiveDataset, get_sets
from model import QClip
from trainer import Trainer

import open_clip

if __name__ == "__main__":
    train_set, test_set = get_sets()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    embedding_width = model.encode_text(tokenizer(["hello", "world"])).shape[1]
    print(f"Embedding width: {embedding_width}")

    train_dataset = SquadLocalContextContrastiveDataset(train_set, model, tokenizer, normalize_clip=True)
    test_dataset = SquadLocalContextContrastiveDataset(test_set, model, tokenizer, normalize_clip=True)

    qclip = QClip(
        embedding_dim = embedding_width,
        output_dim = 512, 
        use_answer_projection = True,
        use_question_projection = True,
        simple_loss = True,
        temperature = 1.0
    )

    trainer = Trainer(
        train_dataset,
        test_dataset,
        qclip,
        batch_size = 32,
        shuffle = True,
    )

    trainer.train(epochs = 1)
