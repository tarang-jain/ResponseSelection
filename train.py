# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertTokenizer
import torch.optim as optim
import pickle
from torch.nn.modules.loss import _Loss
import numpy as np
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of training epochs")
opt = parser.parse_args()

answer_embedding_dict = pickle.load(open("embeddings_a.pkl", "rb+"))

#Defining dataset class
class CornellMovieDialogsDataset(Dataset):

    def __init__(self, question_text_filepath, answer_embedding_dict, maxlen):

        self.questions = open(question_text_filepath, "r").readlines()[:-1]
        self.answer_embedding_dict = answer_embedding_dict
        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):

        question = self.questions[index]
        answer_embedding = (torch.tensor(self.answer_embedding_dict[index])).cuda()

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(question) #Tokenize the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids).cuda() #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = ((tokens_ids_tensor != 0).long()).cuda()

        return tokens_ids_tensor, attn_mask, answer_embedding

#Creating instances of training and validation set
dataset = CornellMovieDialogsDataset(question_text_filepath = 'cornell movie-dialogs corpus/preprocessed_q_train.txt', answer_embedding_dict = answer_embedding_dict, maxlen = 30)

train_set, val_set = torch.utils.data.random_split(dataset, [210000, len(dataset) - 210000])

#Creating intsances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size = 256, shuffle = True)
val_loader = DataLoader(val_set, batch_size = 256, shuffle = True)

class DiscriminativeLoss(_Loss):

    def __init__(self):
        super(DiscriminativeLoss, self).__init__()

    def forward(self, model_output_tensor, answer_embedding_tensor, all_answer_embeddings):
        total_loss = 0.0
        bs, dim = model_output_tensor.size()
        total_len = len(all_answer_embeddings)

        for i in range(bs):
            num = nn.MSELoss()(model_output_tensor[i], answer_embedding_tensor[i])
            print(num)
            den = nn.MSELoss()((model_output_tensor[i]).repeat(1,total_len).view(-1, dim), all_answer_embeddings)
            # print(den)
            total_loss += den
        return total_loss

class ResponseSelector(nn.Module):

    def __init__(self, freeze_bert = True):
        super(ResponseSelector, self).__init__()
        #Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.affine = nn.Linear(768, 1024)

        self.bottleneck = nn.Linear(1024, 128)

        self.final = nn.Linear(128, 768)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        answer_vec = self.affine(cls_rep)
        answer_vec = self.bottleneck(answer_vec)
        answer_vec = self.final(answer_vec)

        return answer_vec

# Commented out IPython magic to ensure Python compatibility.
model = ResponseSelector(freeze_bert = True).cuda()
objf = (nn.MSELoss()).cuda()
optimizer = optim.Adam(model.parameters(),'lr'==0.01)

print("Model, objf, optimizer initialised")

def train_model(train_dataloader, val_dataloader, training = True, epochs=20):
    old_val_loss = 1000000.0
    for epoch in range(epochs):
        model.train()
        loss = 0
        for i, (tokens_ids_tensor, attn_mask, answer_embedding) in enumerate(train_dataloader):
            if training:
                model_output = model(tokens_ids_tensor, attn_mask)
            else:
                with torch.no_grad():
                    model_output= model(tokens_ids_tensor, attn_mask)
            loss = objf(model_output.float(), answer_embedding.float())
            
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if i  % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [MSE loss: %f]"
                     % (epoch, epochs, i, len(train_dataloader), loss.item())
                     )
            model.eval()
            val_loss = 0
            for i, (tokens_ids_tensor, attn_mask, answer_embedding) in enumerate(val_dataloader):
                model_output = model(tokens_ids_tensor, attn_mask)
                val_loss += objf(model_output.float(), answer_embedding.float())

        print(
             "[Epoch %d/%d] [Validation MSE loss: %f]"
             % (epoch, epochs, val_loss.item())
             )
        torch.save(model.state_dict(), "epoch-{}.pt".format(epoch))

        if val_loss < old_val_loss:
            torch.save(model.state_dict(), "best-model.pt")
            old_val_loss = val_loss

print("Start training")

if __name__ == "__main__":
    train_model(train_loader, val_loader, epochs = opt.n_epochs)

