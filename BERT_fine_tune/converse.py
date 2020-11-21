from train import ResponseSelector, CornellMovieDialogsDataset
import os
import pickle
import numpy as np
import tqdm
import ngtpy
import re
import argparse
import torch

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(corpus_name)
a_file = os.path.join(corpus, "preprocessed_a.txt")
a_embedding_file = os.path.join("embeddings_a.pkl")
model_path = os.path.join("best-model.pt")

answer_embedding_dict = pickle.load(open(a_embedding_file, "rb+"))

dataset = CornellMovieDialogsDataset(question_text_filepath = 'cornell movie-dialogs corpus/preprocessed_q_train.txt', answer_embedding_dict = answer_embedding_dict, maxlen = 30)
tokenizer = dataset.tokenizer
max_len = dataset.maxlen
ngtpy.create(b"answers_index", 768)
index = ngtpy.Index(b"answers_index")
all_answer_embeddings = list(answer_embedding_dict.values())
index.batch_insert(all_answer_embeddings)
index.save()

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?,])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?,\']+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

answers = open(a_file, "r").readlines()

model = ResponseSelector()
model.load_state_dict(torch.load(model_path))
model.eval()

while(True):
    query = normalizeString(input("user-input >> "))
    query = "[CLS] " + query + " [SEP]"
    tokens = tokenizer.tokenize(query) #Tokenize the sentence
    if len(tokens) < max_len:
        tokens = tokens + ['[PAD]' for _ in range(max_len - len(tokens))]
    else:
        tokens = tokens[:(max_len-1)] + ['[SEP]']
    
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_ids_tensor = (torch.tensor(tokens_ids)).unsqueeze(0)
    attn_mask = ((tokens_ids_tensor != 0).long()).unsqueeze(0)
    query_embedding = (model(tokens_ids_tensor, attn_mask)).detach().numpy()
    result = index.search(query_embedding, 1)
    
    for i, o in enumerate(result):
        answer_string = answers[int(str(o[0]))]
        answer = (answer_string.strip("[CLS] ")).strip(" [SEP]")
        print(answer)

