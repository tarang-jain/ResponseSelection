# ResponseSelection

This repository consists of the work done during my internship at Sony Corporation, Japan. The aim was to develop a response selection based general purpose question answering system using bert-as-service. I have implemented two approaches for the response selection task, which make use of [bert-as-service](https://github.com/hanxiao/bert-as-service) and [NGT](https://github.com/yahoojapan/NGT) for a quick nearest neighbour search on the embedding vectors. 

## Approach 1: No fine-tuning required

To run it from start to end, the Cornell Movie Dialogs Corpus must be placed in the project directory. Run bert-as-service in a new terminal. All the flags in run.sh should be set to 1. Then run ./run.sh. This will preprocess the corpus, create files of embedding vectors using bert-as-service and run the question answering system. Evaluation is done by setting aside 5000 <question, answer> pairs as the evaluation set. Evaluation questions are passed as input and the cosine similarity between the true response from the eval set and the response obtained from the model is calculated. 

After the neural network is trained, one can converse with the system. This is done through converse.py.
