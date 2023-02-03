import argparse
import torch
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from model import QClip
from config import Config
import open_clip

class ContextSearcher:
    def __init__(self, qclip: QClip, clip_model, clip_tokenizer, context, device: torch.device):
        self.qclip = qclip
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.context = context
        self.device = device

        print("Splitting context into sentences...")
        self.sentences = self.split_context(context)
        print("Computing embeddings...")
        self.context_embeddings = self.compute_embeddings(self.sentences)
        print(f"Done. Context embeddings shape: {self.context_embeddings.shape}")

    def split_context(self, context):
        """
        Splits the context into a list of sentences
        """
        return sent_tokenize(context)

    def compute_embeddings(self, sentences):
        """
        Computes the embeddings for the context. Returns a ndarray of shape (n_sentences, embedding_dim) with the same order as the input
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            tokenized = self.clip_tokenizer(sentences).to(self.device)
            embeddings = self.clip_model.encode_text(tokenized)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            if self.qclip.use_question_projection:
                embeddings = self.qclip.q_projection(embeddings)
        return embeddings.float().detach().cpu().numpy()

    def search(self, query, n_results=10):
        """
        Searches the context for the given query. Returns a list of (sentence, score) tuples
        """
        with torch.no_grad():
            tokenized_query = self.clip_tokenizer([query]).to(self.device)
            query_embedding = self.clip_model.encode_text(tokenized_query)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
            if self.qclip.use_question_projection:
                query_embedding = self.qclip.q_projection(query_embedding)
            query_embedding = query_embedding.float().detach().cpu().numpy()
        
        scores = np.dot(self.context_embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:n_results]

        return [(self.sentences[i], scores[i]) for i in top_indices]
    
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
    
def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = Config(**checkpoint["config"])
    qclip, clip_model = generate_model(config, device)
    qclip.load_state_dict(checkpoint["model"])
    qclip.eval()
    qclip.to(device)

    tokenizer = open_clip.get_tokenizer(config.data.clip_arch)
    
    return qclip, clip_model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--context-file", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    qclip, clip_model, clip_tokenizer = load_checkpoint(args.checkpoint, device)
    searcher = ContextSearcher(qclip, clip_model, clip_tokenizer, open(args.context_file).read(), device)

    inp = ""
    while inp != "exit":
        inp = input("Enter query: ")
        results = searcher.search(inp, n_results=5)
        print("\n------------------")
        for result in results:
            print(result[0])
            print(result[1])
            print("")
        print("------------------\n")

if __name__ == "__main__":
    main()