import gin
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import time



@gin.configurable
class Train():
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion: nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 epochs: int,
                 device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device

    def train(self):
        print("""-------------Training--------------""")
        for epoch in range(self.epochs):
            total_loss=0
            print('----------------epoch {}----------------'.format(epoch))

            for u_id, pos_item, neg_item in self.dataloader:
                u_embeds, pos_i_embeds, neg_i_embeds = self.model(u_id, pos_item, neg_item)

                self.optimizer.zero_grad()
                loss = self.criterion(u_embeds, pos_i_embeds, neg_i_embeds)
                loss.backward()
                self.optimizer.step()
                total_loss += loss

                # print('--------------loss--------------')
                # print('loss:{}'.format(loss))

            print('--------------epoch loss--------------')
            print('epoch loss: {}'.format(total_loss/len(self.dataloader)))


class Test():
    def __init__(self,
                 model: nn.Module,
                 metric,
                 dataloader: torch.utils.data.DataLoader,
                 epochs: int,
                 device='cpu'):
        self.model = model
        self.metric = metric
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device

    def eval(self):
        print("""-------------Evaluation--------------""")
        for epoch in range(self.epochs):
            for u_id, pos_item, neg_item in self.dataloader:
                with torch.no_grad():
                    u_embeds, pos_i_embeds, neg_i_embeds = self.model(u_id, pos_item, neg_item)

                    print('--------------shapes--------------')
                    print(u_id.shape)
                    print(pos_item.shape)
                    print(neg_item.shape)
                    print(u_embeds.shape)
                    print(pos_i_embeds.shape)
                    print(neg_i_embeds.shape)

                    print('--------------loss--------------')
                    print('loss:{}'.format("""loss"""))
