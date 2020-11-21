from bert_serving.client import BertClient
import numpy as np
import os
import argparse
import pickle
import ngtpy
import tqdm
from scipy import spatial

parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=int, default=50, help="number of best questions")
opt = parser.parse_args()

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(corpus_name)
q_file = os.path.join(corpus, "preprocessed_q_train.txt")
qa_file = os.path.join(corpus, "preprocessed_qa_train.txt")
eval_q_file = os.path.join(corpus, "preprocessed_q_eval.txt")
eval_qa_file = os.path.join(corpus, "preprocessed_qa_eval.txt")

q_embedding_file = os.path.join("embeddings_q.pkl")
qa_embedding_file = os.path.join("embeddings_qa.pkl")

bc = BertClient(check_length = False)

f = open(q_embedding_file, "rb+")
all_question_embedding = list((pickle.load(f)).values())
f.close()

train_question_embedding = all_question_embedding[:-5000]
eval_question_embedding = all_question_embedding[-5000:]

f = open(qa_embedding_file, "rb+")
all_question_answer_embedding = list((pickle.load(f)).values())
f.close()

train_qa_embedding = all_question_answer_embedding[:-5000]
eval_qa_embedding = all_question_answer_embedding[-5000:]

ngtpy.create(b"evaluation_questions_index", len(all_question_embedding[0]))
index = ngtpy.Index(b"evaluation_questions_index")
index.batch_insert(train_question_embedding)
index.save()
total_cosine_similarity = 0.0

QA = open(qa_file, "r").readlines()
eval_qa_pairs = open(eval_qa_file, "r").readlines()

eval_questions = open(eval_q_file, "r").readlines()


print("Start evaluation")

for i in tqdm.tqdm(range(len(eval_question_embedding))):
    question = eval_questions[i]
    q_embedding = eval_question_embedding[i]
    result = index.search(q_embedding, opt.beam)
    answers = []
    distances = []
    true_answer = ((eval_qa_pairs[i]).split(" [SEP] "))[1]
    true_answer_embedding = bc.encode([true_answer])
    for i, o in enumerate(result):
        answer = (QA[int(str(o[0]))].split(" [SEP] "))[1]
        answers.append(answer)
        qa_pair = [question + " [SEP] " + answer]
        distances.append(np.linalg.norm(bc.encode(qa_pair) - train_qa_embedding[int(str(o[0]))]))
    best_answer = answers[distances.index(min(distances))]
    best_answer_embedding = bc.encode([best_answer])
    total_cosine_similarity += 1 - spatial.distance.cosine(true_answer_embedding, best_answer_embedding)

    if(i % 100) == 0:
        print("total_cosine_similarity until now", total_cosine_similarity)

total_average_cosine_similarity = total_cosine_similarity/len(eval_questions)
print("total average cosine similarity on eval set:", total_average_cosine_similarity)
