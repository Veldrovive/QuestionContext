from torch.utils.data import Dataset
import open_clip
import torch

from pathlib import Path
import requests
import json
import re
import random
from typing import List, Dict, Tuple

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

SQUAD_TRAIN_SET = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_TEST_SET = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
QUAC_TRAIN_SET = "https://s3.amazonaws.com/my89public/quac/train_v0.2.json"
QUAC_TEST_SET = "https://s3.amazonaws.com/my89public/quac/val_v0.2.json"
CACHE_DIR = Path("/tmp/cli_question_answering/")

def get_squad_sets() -> Tuple[Path, Path]:
    # If the dataset is not in the cache, download it
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True)
    
    train_set = CACHE_DIR / "squad-train-v2.0.json"
    test_set = CACHE_DIR / "squad-dev-v2.0.json"

    if not train_set.exists():
        with open(train_set, "w") as f:
            f.write(requests.get(SQUAD_TRAIN_SET).text)
    if not test_set.exists():
        with open(test_set, "w") as f:
            f.write(requests.get(SQUAD_TEST_SET).text)

    return train_set, test_set

def load_squad_set(path: Path) -> List[Dict]:
    with open(path) as f:
        dataset = json.load(f)
        return dataset["data"]

def get_quac_sets() -> Tuple[Path, Path]:
    # If the dataset is not in the cache, download it
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True)
    
    train_set = CACHE_DIR / "quac-train-v0.2.json"
    test_set = CACHE_DIR / "quac-dev-v0.2.json"

    if not train_set.exists():
        with open(train_set, "w") as f:
            f.write(requests.get(QUAC_TRAIN_SET).text)
    if not test_set.exists():
        with open(test_set, "w") as f:
            f.write(requests.get(QUAC_TEST_SET).text)

    return train_set, test_set

def load_quac_set(path: Path) -> List[Dict]:
    with open(path) as f:
        dataset = json.load(f)
        return dataset["data"]

"""
For reference, the squad dataset is structured as follows:
{
    "version": "string",
    "data": [
        {
            "title": "string",
            "paragraphs": [
                {
                    "context": "string",
                    "qas": [
                        {
                            "id": "string",
                            "question": "string",
                            "answers": [
                                {
                                    "text": "string",
                                    "answer_start": int
                                }
                            ],
                            "is_impossible": bool
                        }
                    ]
                }
            ]
        }
    ]
}
"""

class SquadLocalContextContrastiveDataset(Dataset):
    """
    Generates pairs of questions and the sentences answers appear in from the squad dataset.

    We only keep questions that are answerable.

    Since we will be doing contrastive learning we also generate negative examples which are sentences where the answer does not appear.
    TODO: Decide if negative examples should only come from outside the article
        Now that I'm thinking about it, it would be way to easy to use keyword matching to select the sentence from the right article so at least some negative
        examples need to come from the same article as the positive example
    """
    def __init__(self, squad_set_path: Path, context_radius: int = 1, clip_model = None, clip_tokenizer = None, normalize_clip = False, clip_device = None, include_article_name: bool = True):
        self.squad_set = load_squad_set(squad_set_path)
        self.context_radius = context_radius
        self.clip_model = clip_model
        self.clip_device = clip_device
        self.clip_tokenizer = clip_tokenizer
        self.normalize_clip = normalize_clip
        self.include_article_name = include_article_name

        self.article_sentences: Dict[str, List[str]] = {}  # Maps from article titles to the sentences in the article
        self.qid_map: Dict[str, Tuple[str, List[str]]] = {}  # Maps question ids to their questions and the sentences the answers appear in
        self.qids: List[str] = []  # List of all question ids
        self.qid_article_map: Dict[str, str] = {}  # Maps question ids to the article they are from
        self.preprocess_data()

    def preprocess_data(self):
        """
        We need two structures:
        A dictionary that maps question ids to questions and the sentences answers appear in
        A list of all sentences in the dataset split by article
        """
        for article in self.squad_set:
            title, paragraphs = article["title"], article["paragraphs"]
            self.article_sentences[title] = []
            for paragraph in paragraphs:
                context, questions = paragraph["context"], paragraph["qas"]
                # Find the indices of all new sentences
                # sentence_indices = [0]
                # for match in re.finditer(r"\. ", context):
                #     sentence_indices.append(match.end())
                # sentence_indices.append(len(context))
                # # Split the context into sentences
                # sentences = [context[i:j] for i, j in zip(sentence_indices, sentence_indices[1:])]
                sentences = sent_tokenize(context)
                sentence_indices = [0]
                for sentence in sentences:
                    sentence_indices.append(sentence_indices[-1] + len(sentence) + 1) # +1 for the space between sentences

                self.article_sentences[title].extend(sentences)
                for question in questions:
                    qid, question_text, answers, is_impossible = question["id"], question["question"], question["answers"], question["is_impossible"]
                    if is_impossible:
                        continue
                    # Find the sentences that contain the answers
                    self.qids.append(qid)
                    self.qid_article_map[qid] = title
                    answer_sentences = []
                    # The same answer may appear multiple times for some reason. We only keep one example of each answer that appears at the same position
                    seen_answers = set()
                    for answer in answers:
                        answer_text, answer_start = answer["text"], answer["answer_start"]
                        for i, sentence in enumerate(sentences):
                            if answer_start >= sentence_indices[i] and answer_start < sentence_indices[i+1]:
                                # To get the context we combine the sentence i as well as the self.context_radius sentences before and after it
                                # If we run out of sentences in either direction we just use as many as we can
                                context = "".join(sentences[max(0, i-self.context_radius):i+1+self.context_radius])

                                if context in seen_answers:
                                    # Each sentence may only be an answer once. The same sentence being an answer to the same question multiple times doesn't make sense
                                    break

                                answer_sentences.append(context)
                                seen_answers.add(context)
                                break
                    if len(answer_sentences) == 0:
                        raise ValueError(f"Could not find sentence for answer {answer_text} in context {context}")
                    if self.include_article_name:
                        article_title = title.replace("_", " ")
                        question_text = f"This is a question about {article_title}. {question_text}"
                    self.qid_map[qid] = (question_text, answer_sentences)

    def __len__(self):
        """
        We may have more than one answer for each question, but if we do we simply select one at random.
        """
        return len(self.qids)

    def __getitem__(self, idx: int):
        """
        Select the question based on the index. If there are multiple possible answer sentences, select one at random.
        If we have a clip model, we return the text features for the question and answer. Otherwise we return the text.
        """
        qid = self.qids[idx]
        question, answer_sentences = self.qid_map[qid]
        # Select a random positive example
        positive_example = random.choice(answer_sentences)
        if self.clip_model is not None:
            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_question, clip_positive_example = self.clip_model.encode_text(
                    self.clip_tokenizer([question, positive_example]).to(self.clip_device)
                ).float()
                if self.normalize_clip:
                    clip_question /= clip_question.norm(dim=-1, keepdim=True)
                    clip_positive_example /= clip_positive_example.norm(dim=-1, keepdim=True)
                return clip_question, clip_positive_example, question, positive_example, qid
        return question, positive_example, qid

class QuacLocalContextContrastiveDataset(Dataset):
    def __init__(self, squad_set_path: Path, context_radius: int = 1, clip_model = None, clip_tokenizer = None, normalize_clip = False, clip_device = None, include_article_name: bool = True):
        """
        A dataset that returns a question and a positive example from the same context.
        The positive example is selected at random from the set of sentences that contain the answer to the question.
        The negative example is selected at random from the set of sentences in the same article that do not contain the answer to the question.
        """
        self.squad_set = load_quac_set(squad_set_path)
        self.context_radius = context_radius
        self.clip_model = clip_model
        self.clip_device = clip_device
        self.clip_tokenizer = clip_tokenizer
        self.normalize_clip = normalize_clip
        self.include_article_name = include_article_name

        self.article_sentences: Dict[str, List[str]] = {}  # Maps from article titles to the sentences in the article
        self.qid_map: Dict[str, Tuple[str, List[str]]] = {}  # Maps question ids to their questions and the sentences the answers appear in
        self.qids: List[str] = []  # List of all question ids
        self.qid_article_map: Dict[str, str] = {}  # Maps question ids to the article they are from
        self.preprocess_data()

    def preprocess_data(self):
        """
        We need two structures:
        A dictionary that maps question ids to questions and the sentences answers appear in
        A list of all sentences in the dataset split by article
        """
        for article in self.squad_set:
            title, section_title, paragraphs = article["title"], article["section_title"], article["paragraphs"]
            self.article_sentences[title] = []
            for paragraph in paragraphs:
                context, questions = paragraph["context"], paragraph["qas"]
                # Find the indices of all new sentences
                # sentence_indices = [0]
                # for match in re.finditer(r"\. ", context):
                #     sentence_indices.append(match.end())
                # sentence_indices.append(len(context))
                # # Split the context into sentences
                # sentences = [context[i:j] for i, j in zip(sentence_indices, sentence_indices[1:])]
                sentences = sent_tokenize(context)
                sentence_indices = [0]
                for sentence in sentences:
                    sentence_indices.append(sentence_indices[-1] + len(sentence) + 1)
                self.article_sentences[title].extend(sentences)
                for question in questions:
                    qid, question_text, answers = question["id"], question["question"], question["answers"]
                    answers = [answer for answer in answers if answer["text"] != "CANNOTANSWER"]
                    if len(answers) == 0:
                        continue
                    self.qids.append(qid)
                    self.qid_article_map[qid] = title
                    answer_sentences = []
                    seen_answers = set()
                    for answer in answers:
                        answer_text, answer_start = answer["text"], answer["answer_start"]
                        for i, sentence in enumerate(sentences):
                            if answer_start >= sentence_indices[i] and answer_start < sentence_indices[i+1]:
                                # To get the context we combine the sentence i as well as the self.context_radius sentences before and after it
                                # If we run out of sentences in either direction we just use as many as we can
                                context = "".join(sentences[max(0, i-self.context_radius):i+1+self.context_radius])

                                if context in seen_answers:
                                    # Each sentence may only be an answer once. The same sentence being an answer to the same question multiple times doesn't make sense
                                    break

                                answer_sentences.append(context)
                                seen_answers.add(context)
                                break
                    if len(answer_sentences) == 0:
                        print(answers)
                        raise ValueError(f"Could not find sentence for answer {answer_text} in context {context}")
                    if self.include_article_name:
                        question_text = f"This is a question about {title}. Specifically, {section_title}. {question_text}"
                    self.qid_map[qid] = (question_text, answer_sentences)

    def __len__(self):
        return len(self.qids)
    
    def __getitem__(self, idx: int):
        """
        Select the question based on the index. If there are multiple possible answer sentences, select one at random.
        If we have a clip model, we return the text features for the question and answer. Otherwise we return the text.
        """
        qid = self.qids[idx]
        question, answer_sentences = self.qid_map[qid]
        # Select a random positive example
        positive_example = random.choice(answer_sentences)
        if self.clip_model is not None:
            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_question, clip_positive_example = self.clip_model.encode_text(
                    self.clip_tokenizer([question, positive_example]).to(self.clip_device)
                ).float()
                if self.normalize_clip:
                    clip_question /= clip_question.norm(dim=-1, keepdim=True)
                    clip_positive_example /= clip_positive_example.norm(dim=-1, keepdim=True)
                return clip_question, clip_positive_example, question, positive_example, qid
        return question, positive_example, qid

if __name__ == "__main__":
    train_set, test_set = get_squad_sets()
    use_clip = False
    if use_clip:
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
        tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
        dataset = SquadLocalContextContrastiveDataset(train_set, clip_model, tokenizer, normalize_clip=True, context_radius=0)
    else:
        dataset = SquadLocalContextContrastiveDataset(train_set, context_radius=0)
    # print(dataset.qid_map)
    # print(dataset.article_sentences)
    # print(len(dataset))
    # print(dataset[0])

    qid = dataset.qids[2000]
    print(dataset.qid_map[qid])

    train_set, test_set = get_quac_sets()
    use_clip = False
    if use_clip:
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
        tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
        dataset = QuacLocalContextContrastiveDataset(train_set, clip_model, tokenizer, normalize_clip=True, context_radius=0)
    else:
        dataset = QuacLocalContextContrastiveDataset(train_set, context_radius=0)
    # print(dataset.qid_map)
    # print(dataset.article_sentences)
    print(len(dataset))

    qid = dataset.qids[5000]
    print(dataset.qid_map[qid])
