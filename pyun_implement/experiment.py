import gin
from torch.utils.data import Dataset
import torch as t
import torch.nn as nn
import numpy as np
import time


@gin.configurable
class Train():
    def __init__(self,
                 model: nn.Module,
                 optimizer: t.optim,
                 criterion: nn.Module,
                 train_dataloader: t.utils.data.DataLoader,
                 test_dataloader: t.utils.data.DataLoader,
                 epochs: int,
                 device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trani_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.device = device

    def train(self):
        for epoch in range(self.epochs):

            t1 = time.time()
            total_loss = 0
            print('------------------------epoch {}------------------------'.format(epoch+1))
            b_i = 0
            for u_id, pos_item, neg_item in self.trani_dataloader:
                t2 = time.time()
                u_embeds, pos_i_embeds, neg_i_embeds = self.model(u_id, pos_item, neg_item, True)

                break
                self.optimizer.zero_grad()
                loss = self.criterion(u_embeds, pos_i_embeds, neg_i_embeds)
                loss.backward()
                self.optimizer.step()
                total_loss += loss

                print('|-------mini batch {} loss-------|'.format(b_i+1))
                print('|loss:{}|'.format((loss)))
                print('|run time:{}|'.format(round(time.time()-t2, 4)))
                b_i += 1

            test = Test(model=self.model,
                        dataloader=self.test_dataloader,
                        ks=10,
                        device='cpu')

            print('|------------------------epoch loss------------------------|')
            print('|epoch loss: {} run time:{}|'.format((total_loss/len(self.trani_dataloader)), round(time.time()-t1, 4)))
            test.eval()


class Test():
    def __init__(self,
                 model: nn.Module,
                 dataloader: t.utils.data.DataLoader,
                 ks: int,
                 device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.ks = ks
        self.device = device


    def eval(self):
        with t.no_grad():
            t2 = time.time()
            eval_score = 0
            for u_id, pos_items in self.dataloader:
                u_id, pos_items = u_id.to(self.device), pos_items.to(self.device)

                u_embeds, pos_i_embeds, _ = self.model(users=u_id,
                                                       pos_items=pos_items,
                                                       neg_items=t.empty(0),
                                                       node_flag=False)

                pred_ratings = t.mm(u_embeds, pos_i_embeds.T)
                _, pred_rank = t.topk(pred_ratings, self.ks)

                gt_rank = pos_items.nonzero(as_tuple=True)[0]

                pred_rank = pred_rank[0]

                pred_score = np.sum(np.abs(self.ks-(pred_rank-gt_rank)).numpy()/np.log2(np.arange(2, len(gt_rank)+2)))
                ideal_score = np.sum(self.ks/np.log2(np.arange(2, len(gt_rank)+2)))

                eval_score += (pred_score / ideal_score)

            print('evaluation score: {}  run time: {}'.format(eval_score/len(self.dataloader), round(time.time()-t2, 4)))
