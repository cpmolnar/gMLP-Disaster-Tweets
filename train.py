from g_mlp_pytorch import gMLP
from g_mlp_pytorch.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import os
import pandas as pd

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
SEQ_LEN = 768

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

# instantiate GPT-like decoder model

model = gMLP(
    num_tokens = 30000,
    dim = 512,
    seq_len = SEQ_LEN,
    depth = 8
)

model = AutoregressiveWrapper(model)
model.cuda()

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

class TextSamplerDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data.reset_index()

    def __getitem__(self, data):
        random_num = np.random.randint(0,len(self.data))
        if self.data.keyword[random_num] is np.nan:
            keyword = ''
        else:
            keyword = self.data.keyword[random_num]
        text = self.data.text[random_num]
        target = self.data.target[random_num]

        return keyword, text, target

    def __len__(self):
        return len(self.data)

df = pd.read_csv('data/train.csv')
data_train = df.sample(frac = 0.7)
data_val = df.drop(data_train.index)

train_dataset = TextSamplerDataset(data_train)
val_dataset   = TextSamplerDataset(data_val)

train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training
pbar = tqdm.tqdm(range(NUM_BATCHES), mininterval=10.)
train_loss=0.0
for i in pbar:
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        keyword, text, target = next(train_loader)
        batch = tokenizer(keyword, text, padding='max_length', max_length=SEQ_LEN)
        input_ids = torch.tensor(batch.data['input_ids']).cuda()
        target = target.cuda()

        pred = model(input_ids)
        loss = F.cross_entropy(pred, target, ignore_index=model.ignore_index)
        loss.backward()

    train_loss+=loss

    pbar.set_description(f"Iteration {i+1}, training_loss: {train_loss/((i%VALIDATE_EVERY)+1)}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == VALIDATE_EVERY-1:
        train_loss=0.0 #reset avg train loss
        model.eval()
        with torch.no_grad():
            val_loss=0.0
            preds_corr=0
            for (keyword, text, target) in val_loader:
                batch = tokenizer(keyword, text, padding='max_length', max_length=SEQ_LEN)
                input_ids = torch.tensor(batch.data['input_ids']).cuda()
                target = target.cuda()

                pred = model(input_ids)
                val_loss += F.cross_entropy(pred, target, ignore_index=model.ignore_index)/len(data_val)
                preds_corr+=(pred.argmax(dim=1)==target).sum()
            print(f'{"*"*100}')
            print(f'validation loss: {val_loss.item()}')
            print(f'validation acc: {preds_corr/len(data_val)}')
            print(f'{"*"*100}')
            print()

            # model.save('model.')