# bert_fine-tuning_response_selection

This repository was created during my internship at Sony Corporation. 

It is a response selection question answering system based on the Cornell Movie Dialogs Corpus. The underlying system makes use of bert-as-service to get sentence embeddings and NGT for quick nearest neighbour search.

To run it from start to end, the Cornell Movie Dialogs Corpus must be placed in the project directory. Run bert-as-service in a new terminal. Then all the flags in run.sh should be set to 1. Then run ./run.sh. This will preprocess the corpus and create files of embedding vectors using bert-as-service. This is followed by training the neural network. The neural network training (bert fine-tuning) script is train.py. It needs a GPU to run the training process.

After the neural network is trained, one can converse with the system. This is done through converse.py.
