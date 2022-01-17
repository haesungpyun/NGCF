import torch
import torch.nn as nn
import numpy as np
import math


class Evaluation():
    def __init__(self,
                 test_dataloader,
                 model,
                 device,
                 top_k: int = 20,
                 ):
        self.dataloader = test_dataloader
        self.model = model
        self.top_k = top_k
        self.device = device

    def Ndcg(self, gt_item, pred_items):
        # IDCG = self.dcg(gt_items)
        # DCG = self.dcg(pred_items)
        # output = DCG/IDCG
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            return np.reciprocal(np.log2(index + 2))
        return 0

    def hit(self, gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

    def get_metric(self):
        NDCG = []
        HR = []
        device = self.device
        self.model.eval()
        with torch.no_grad():
            for users, pos_items in self.dataloader:
                users, pos_items = users.to(device), pos_items.to(device)

                user_embeddings, pos_item_embeddings, _ = self.model(users=users,
                                                                     pos_items=pos_items,
                                                                     neg_items=[],
                                                                     use_dropout=False)


                pred_matrix = torch.matmul(user_embeddings, torch.transpose(pos_item_embeddings, 0, 1))

                _, pred_indices = torch.topk(pred_matrix[0], self.top_k)

                recommends = torch.take(
                    pred_matrix[0], pred_indices).cpu().numpy().tolist()

                # _,gt_indices=torch.topk(trained_matrix[users[0]],self.top_k)
                #
                # ground_truth = torch.take(
                #     trained_matrix[users[0]],gt_indices).cpu().numpy().tolist()
                # gt_item = pos_items[0].item()
                HR.append(self.hit(gt_item=gt_item, pred_items=recommends))
                NDCG.append(self.Ndcg(gt_item=gt_item, pred_items=recommends))

        return np.mean(HR), np.mean(NDCG)