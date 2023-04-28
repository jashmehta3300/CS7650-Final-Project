import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    AdamW
)
from data import *
from utils import *
from config import args

class Client:

    def __init__(self, args, epochs, data, idxs, tokenizer):
        self.args = args
        self.epochs = epochs
        self.dataloader = DataLoader(
            CustomDataset(data, idxs),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
        )

    def train(self, model):

        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.args.class_weights_array).cuda()
        )
        optimizer = AdamW(
            model.parameters(),
            lr=self.args.client_lr,
            weight_decay = 0,
        )
        global_model = copy.deepcopy(model)
        model.train()

        epoch_loss = []
        for _ in range(1, self.epochs + 1):
            batch_loss = []
            for batch in self.dataloader:
                batch = {k: v.cuda() for k, v in batch.items()}

                optimizer.zero_grad()
                output = model(**batch)
                logits = output.logits

                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                
                if int(self.args.K) == 1:
                    loss = criterion(logits, batch["labels"])
                else:
                    loss = criterion(logits, batch["labels"]) + (self.args.mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        total_loss = sum(epoch_loss) / len(epoch_loss)
    
        return model.state_dict(), total_loss