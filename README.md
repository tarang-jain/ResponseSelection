# bert_fine-tuning_response_selection

This repository was created during my internship at Sony Corporation. 

It is a response selection question answering system based on the Cornell Movie Dialogs Corpus. The underlying system makes use of bert-as-service to get sentence embeddings and NGT for quick nearest neighbour search.

To run it from start to end, the Cornell Movie Dialogs Corpus must be placed in the project directory. Run bert-as-service in a new terminal. Then all the flags in run.sh should be set to 1. Then run ./run.sh. This will preprocess the corpus, create files of embedding vectors using bert-as-service and then run it will run the question answering system. Evaluation is done by setting aside 5000 <question, answer> pairs as the evaluation set. Evaluation questions are passed as input and the cosine similarity between the true response from the eval set and the response obtained from the model is calculated. 
