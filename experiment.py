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

                break

                self.optimizer.zero_grad()
                loss = self.criterion(u_embeds, pos_i_embeds, neg_i_embeds)
                loss.backward()
                self.optimizer.step()
                total_loss += loss

                print('|-------mini batch {} loss-------|'.format(b_i))
                print('|loss:{}|'.format((loss)))
                print('|run time:{}|'.format(round(time.time()-t2, 4)))
                b_i += 1
            break
            print('|------------------------epoch loss------------------------|')
            print('|epoch loss: {}|'.format((total_loss/len(self.dataloader))))
            print('|run time:{}|'.format(round(time.time()-t1, 4)))


class Test():
    def __init__(self,
                 model: nn.Module,
                 dataframe: pd.DataFrame,
                 dataloader: t.utils.data.DataLoader,
                 epochs: int,
                 ks: int,
                 device='cpu'):
        self.model = model
        self.dataframe = dataframe
        self.dataloader = dataloader
        self.epochs = epochs
        self.ks = ks
        self.device = device

    def eval(self):
        print("""-------------Evaluation--------------""")
        with t.no_grad():
            for u_id, pos_item in self.dataloader:
                u_embeds, pos_i_embeds, _ = self.model(u_id, pos_item, t.empty(0), False)

                print(self.model.u_g_embeddings.shape, self.model.i_g_embeddings.t().shape)
                scores = t.mm(self.model.u_g_embeddings, self.model.i_g_embeddings.t())
                _, pred_idx = t.topk(pos_i_embeds, self.ks)

                recommends = t.take(pos_item, pred_idx)








    def compute_ndcg_k(self, pred_items, test_items, test_indices, k):
        """
        Compute NDCG@k

        Arguments:
        ---------
        pred_items: binary tensor with 1s in those locations corresponding to the predicted item interactions
        test_items: binary tensor with 1s in locations corresponding to the real test interactions
        test_indices: tensor with the location of the top-k predicted items
        k: k'th-order
        Returns:
        -------
        NDCG@k
        Fork from https://github.com/metahexane/ngcf_pytorch_g61/blob/master/utils/helper_functions.py
        """
        r = (test_items * pred_items).gather(1, test_indices)
        f = t.from_numpy(np.log2(np.arange(2, k + 2))).float().cuda()
        dcg = (r[:, :k] / f).sum(1)
        dcg_max = (t.sort(r, dim=1, descending=True)[0][:, :k] / f).sum(1)
        ndcg = dcg / dcg_max
        ndcg[t.isnan(ndcg)] = 0
        return ndcg