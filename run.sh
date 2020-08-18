#!/bin/bash

#Flags
preprocess=0               #set to 1 if data processing needed
encode_data=0              #set to 1 if embeddings are not already created and saved
run_question_answering=1   #set to 1 to converse with the model
run_eval=1                 #set to 1 to run evaluation

question_beam=50 #number of <question, answer> pairs to be searched

printf "\n"
echo "Starting bert service in new terminal window ..."

#if you have a bert service already running, you should first close it

gnome-terminal -e "bash -c \"bert-serving-start -model_dir ../BertModels/uncased_L-12_H-768_A-12 -num_worker 1 -max_seq_len 40; exec bash\""
#set the model_dir as the directory where your bert model is placed

if [ $preprocess -eq 1 ]; then
printf "\n"
echo ============================================================================
echo "                          Preprocessing Corpus                            "
echo ============================================================================

python3 preprocess.py

mv "cornell movie-dialogs corpus" "cornell_movie-dialogs_corpus"
corpus_dir="cornell_movie-dialogs_corpus"

#Check whether eval files are created; if not create them
if [ ! -f $corpus_dir/preprocessed_q_eval.txt ]; then
	printf "\n"
	echo "Preparing eval questions ..."
	tail -n 11283 $corpus_dir/preprocessed_q.txt > $corpus_dir/preprocessed_q_eval.txt
fi

if [ ! -f $corpus_dir/preprocessed_qa_eval.txt ]; then
        printf "\n"
        echo "Preparing eval <question, answer> pairs ..."
        tail -n 11283 $corpus_dir/preprocessed_qa.txt > $corpus_dir/preprocessed_qa_eval.txt
fi

mv "cornell_movie-dialogs_corpus" "cornell movie-dialogs corpus"

fi

if [ $encode_data -eq 1 ]; then
printf "\n"
echo ============================================================================
echo "         Embedding movie lines as vectors using bert-as-service           "
echo ============================================================================

python3 encode_data.py

fi

if [ $run_question_answering -eq 1 ]; then
printf "\n"
echo ============================================================================
echo "            Run question answering using 'question beam                   "
echo ============================================================================

python3 converse.py --beam=$question_beam

fi

if [ $run_eval -eq 1 ]; then
printf "\n"
echo ============================================================================
echo "                         Cosine Similarity Test                           "
echo ============================================================================

python3 eval.py --beam=$question_beam

fi
