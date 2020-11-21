import os
import pickle
import numpy as np
import tqdm
from bert_serving.client import BertClient

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(corpus_name)
q_file = os.path.join(corpus, "preprocessed_q.txt")
qa_file = os.path.join(corpus, "preprocessed_qa.txt")
q_embedding_file = os.path.join("embeddings_q.pkl")
qa_embedding_file = os.path.join("embeddings_qa.pkl")

def save_embeddings(client, datafile, embedding_file):
    embeddings = dict()
    f = open(embedding_file, "wb+")
    Lines = open(datafile, 'r').readlines()

    for i in tqdm.tqdm(range(len(Lines))):
        embeddings[i] = ((client.encode([Lines[i].strip("\n")])).squeeze()).tolist()

    pickle.dump(embeddings, f)
    return

if __name__ == "__main__":

    bc = BertClient(check_length=False)

    print("\nWriting embeddings for questions to embeddings_q.pkl ...")
    
    save_embeddings(bc, q_file, q_embedding_file)
    
    print("\nWriting embeddings for <question, answer> pairs to embeddings_qa.pkl ...")
    
    save_embeddings(bc, qa_file, qa_embedding_file)
