import gin
from torch.utils.data import Dataset
import torch as t
import torch.nn as nn
import evaluation
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import time
from sklearn.metrics import ndcg_score


@gin.configurable
class Train():
    def __init__(self,
                 model: nn.Module,
                 optimizer: t.optim,
                 criterion: nn.Module,
                 dataloader: t.utils.data.DataLoader,
                 epochs: int,
                 device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device

    def train(self):
        print("===============================Training===============================")
        for epoch in range(self.epochs):

            t1 = time.time()
            total_loss = 0
            print('------------------------epoch {}------------------------'.format(epoch))
            b_i = 0
            for u_id, pos_item, neg_item in self.dataloader:
                t2 = time.time()
                u_embeds, pos_i_embeds, neg_i_embeds = self.model(u_id, pos_item, neg_item, True)

                self.optimizer.zero_grad()
                loss = self.criterion(u_embeds, pos_i_embeds, neg_i_embeds)
                loss.backward()
                self.optimizer.step()
                total_loss += loss

                print('|-------mini batch {} loss-------|'.format(b_i))
                print('|loss:{}|'.format((loss)))
                print('|run time:{}|'.format(round(time.time()-t2, 4)))
                b_i += 1

            print('|------------------------epoch loss------------------------|')
            print('|epoch loss: {}|'.format((total_loss/len(self.dataloader))))
            print('|run time:{}|'.format(round(time.time()-t1, 4)))


class Test():
    def __init__(self,
                 model: nn.Module,
                 dataframe: pd.DataFrame,
                 dataloader: t.utils.data.DataLoader,
                 ks: int,
                 device='cpu'):
        self.model = model
        self.dataframe = dataframe
        self.dataloader = dataloader
        self.ks = ks
        self.device = device


    def eval(self):
        print("""-------------Evaluation--------------""")
        with t.no_grad():
            for u_id, pos_items in self.dataloader:
                u_id, pos_items = u_id.to(self.device), pos_items.to(self.device)

                u_embeds, pos_i_embeds, _ = self.model(users=u_id,
                                                       pos_items=pos_items,
                                                       neg_items=t.empty(0),
                                                       node_flag=False)

                pred_ratings = t.mm(u_embeds, pos_i_embeds.T)
                _, pred_rank = t.topk(pred_ratings, self.ks)

                gt_rank = pos_items.nonzero(as_tuple=True)[0]

                my_metric = np.sum(np.abs(pred_rank-gt_rank).numpy()/np.log2(np.arange(len(gt_rank)+2, 2, -1)))

                #ndcg = ndcg_score(gt_rank.numpy(), pred_rank.numpy())

                print('my_metric:', my_metric)
