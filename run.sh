#!/bin/bash

#Flags
preprocess=1               #set to 1 if data processing needed
encode_data=1              #set to 1 if embeddings are not already created and saved
run_question_answering=1   #set to 1 to converse with the model

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
