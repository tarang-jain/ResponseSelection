import numpy as np
import os
from bert_serving.client import BertClient
import argparse
import pickle
import ngtpy
from scipy import spatial

parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=int, default=50, help="number of best questions")
opt = parser.parse_args()

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(corpus_name)
q_file = os.path.join(corpus, "preprocessed_q.txt")
qa_file = os.path.join(corpus, "preprocessed_qa.txt")
eval_q_file = os.path.join(corpus, "preprocessed_q_eval.txt")
eval_qa_file = os.path.join(corpus, "preprocessed_qa_eval.txt")

q_embedding_file = os.path.join("embeddings_q.pkl")
qa_embedding_file = os.path.join("embeddings_qa.pkl")

bc = BertClient(check_length = False)
f = open(q_embedding_file, "rb+")
all_question_embedding = list((pickle.load(f)).values())
f.close()
f = open(qa_embedding_file, "rb+")
all_question_answer_embedding = list((pickle.load(f)).values())
f.close()
index = ngtpy.Index(b"questions_index")
QA = open(qa_file, "r").readlines()
eval_questions = open(eval_q_file, "r").readlines()
eval_qa_pairs = open(eval_qa_file, "r").readlines()
total_cosine_similarity = 0.0

for i in range(len(eval_questions)):
    question = eval_questions[i]
    q_embedding = list(bc.encode([question]))
    result = index.search(q_embedding, opt.beam)
    answers = []
    distances = []
    true_answer = ((eval_qa_pairs[i]).split(" [SEP] "))[1]
    true_answer_embedding = bc.encode([true_answer])
    for i, o in enumerate(result):
        answer = (QA[int(str(o[0]))].split(" [SEP] "))[1]
        answers.append(answer)
        qa_pair = [question + " [SEP] " + answer]
        distances.append(np.linalg.norm(bc.encode(qa_pair) - all_question_answer_embedding[int(str(o[0]))]))
    best_answer = answers[distances.index(min(distances))]
    best_answer_embedding = bc.encode([best_answer])
    total_cosine_similarity += 1 - spatial.distance.cosine(true_answer_embedding, best_answer_embedding)

total_average_cosine_similarity = total_cosine_similarity/len(eval_questions)
print("total average cosine similarity on eval set:", total_average_cosine_similarity)
