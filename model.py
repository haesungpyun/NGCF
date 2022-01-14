import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import gin
from utils import MovieLens
from utils import Download
from matrix import Matrix
import torch


@gin.configurable()  # model에 인풋 넣을 떼 n_users, n_itmes에 1 더하기!
class NGCF(nn.Module):
    def __init__(self,
                 n_user: int,
                 n_item: int,
                 embed_size: int,
                 layer_size: list,
                 node_dropout: float,
                 mess_dropout: list,
                 lap_mat: t.sparse.FloatTensor,
                 eye_mat: t.sparse.FloatTensor,
                 device='cpu'):
        super(NGCF, self).__init__()

        self.n_user = n_user
        self.n_item = n_item

        self.emb_size = embed_size
        self.weight_size = layer_size
        self.n_layer = len(self.weight_size)

        self.device = device

        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        # self.user_embedding = nn.Parameter(t.randn(self.n_user, self.emb_size))
        # self.item_embedding = nn.Parameter(t.randn(self.n_item, self.emb_size))
        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_item, self.emb_size)

        self.w1_list = nn.ModuleList()
        self.w2_list = nn.ModuleList()

        self.node_dropout_list = nn.ModuleList()
        self.mess_dropout_list = nn.ModuleList()

        self.lap_mat = lap_mat
        self.eye_mat = eye_mat

        self.set_layers()

    def set_layers(self):

        initializer = nn.init.xavier_uniform_

        # initial embedding layer
        initializer(self.user_embedding.weight)
        initializer(self.item_embedding.weight)

        weight_size_list = [self.emb_size] + self.weight_size

        # set propagation layer
        for k in range(self.n_layer):
            # set W1, W2 as Linear layer
            self.w1_list.append(
                nn.Linear(in_features=weight_size_list[k], out_features=weight_size_list[k + 1], bias=True))
            self.w2_list.append(
                nn.Linear(in_features=weight_size_list[k], out_features=weight_size_list[k + 1], bias=True))

            # node_dropout on Laplacian matrix
            if self.node_dropout is not None:
                self.node_dropout_list.append(nn.Dropout(p=self.node_dropout))

            # mess_dropout on l-th message
            if self.mess_dropout is not None:
                self.mess_dropout_list.append(nn.Dropout(p=self.mess_dropout[k]))

    def sparse_dropout(self):
        node_mask = nn.Dropout(self.node_dropout)(t.tensor(np.ones(self.lap_mat._nnz()))).type(t.bool)
        i = self.lap_mat._indices()
        v = self.lap_mat._values()
        i = i[:, node_mask]
        v = v[node_mask]

        drop_lap_mat = t.sparse.FloatTensor(i, v, self.lap_mat.shape).to(self.lap_mat.device)
        return drop_lap_mat

    def forward(self, users, pos_items, neg_items):

        ego_embedding = t.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embedding = [ego_embedding]

        for i in range(self.n_layer):
            # node dropout laplacian matrix
            drop_lap_mat = self.sparse_dropout()

            L_I_E = t.mm(drop_lap_mat + self.eye_mat, ego_embedding)
            L_I_E_W1 = self.w1_list[i](L_I_E)

            L_E = t.mm(drop_lap_mat, ego_embedding)
            L_E_E = L_E * ego_embedding
            L_I_E_W2 = self.w2_list[i](L_E_E)

            message_embedding = L_I_E_W1 + L_I_E_W2

            ego_embedding = nn.LeakyReLU(negative_slope=0.2)(message_embedding)

            ego_embedding = self.mess_dropout_list[i](ego_embedding)

            norm_embedding = F.normalize(ego_embedding, p=2, dim=1)

            all_embedding += [norm_embedding]

        all_embedding = t.cat(all_embedding, dim=1)
        u_g_embeddings = all_embedding[:self.n_user, :]
        i_g_embeddings = all_embedding[self.n_user:, :]

        u_g_embeddings = u_g_embeddings[users - 1, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items - 1, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items - 1, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
