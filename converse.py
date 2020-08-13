import os
import pickle
import numpy as np
import tqdm
import ngtpy
from bert_serving.client import BertClient
import re
import argparse
# from preprocess import normalizeString

parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=int, default=50, help="number of best questions")
opt = parser.parse_args()

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(corpus_name)
q_file = os.path.join(corpus, "preprocessed_q.txt")
qa_file = os.path.join(corpus, "preprocessed_qa.txt")
q_embedding_file = os.path.join("embeddings_q.pkl")
qa_embedding_file = os.path.join("embeddings_qa.pkl")

bc = BertClient(check_length = False)
f = open(q_embedding_file, "rb+")
all_question_embedding = list((pickle.load(f)).values())
f.close()
f = open(qa_embedding_file, "rb+")
all_question_answer_embedding = list((pickle.load(f)).values())
f.close()

ngtpy.create(b"questions_index", len(all_question_embedding[0]))
index = ngtpy.Index(b"questions_index")
index.batch_insert(all_question_embedding)
index.save()

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?,])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?,\']+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

QA = open(qa_file, "r").readlines()
while(True):
    query = input("user-input>>> ")
    query = ["CLS " + normalizeString(query)]
    query_embedding = list(bc.encode(query))
    result = index.search(query_embedding, opt.beam)
    answers = []
    distances = []
    for i, o in enumerate(result):
        answer = (QA[int(str(o[0]))].split(" [SEP] "))[1]
        answers.append(answer)
        query_answer_pair = [query[0] + " [SEP] " + answer]
        distances.append(np.linalg.norm(bc.encode(query_answer_pair) - all_question_answer_embedding[int(str(o[0]))]))
    print(answers[distances.index(min(distances))])
