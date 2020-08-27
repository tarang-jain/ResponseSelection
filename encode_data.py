import os
import pickle
import numpy as np
import tqdm
from bert_serving.client import BertClient

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(corpus_name)
a_file = os.path.join(corpus, "preprocessed_a_train.txt")
a_embedding_file = os.path.join("embeddings_a.pkl")

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

    print("\nWriting embeddings for answers to embeddings_a.pkl ...")

    save_embeddings(bc, a_file, a_embedding_file)
