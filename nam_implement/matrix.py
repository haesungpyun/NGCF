import pandas as pd
import torch as t
import scipy.sparse as sp
import numpy as np


class Matrix(object):
    """
    Manage all operations according to Matrix creation
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_user = df['userId'].max()
        self.n_item = df['movieId'].max()

        self.R = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        self.adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.laplacian_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.sparse_norm_adj = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
        self.eye_mat = sp.dok_matrix((self.sparse_norm_adj.shape[0], self.sparse_norm_adj.shape[0]), dtype=np.float32)

    def create_matrix(self):
        u_m_set = tuple(zip(self.df['userId'], self.df['movieId']))
        for u, m in u_m_set:
            self.R[u-1, m-1] = 1.

        # A = [[0, R],[R.T,0]]
        adj_mat = self.adj_mat.tolil()
        R = self.R.tolil()
        adj_mat[:self.n_user, self.n_user:] = R
        adj_mat[self.n_user:, :self.n_user] = R.T
        self.adj_mat = adj_mat.todok()

        # L = D^-1/2 * A * D^-1/2
        diag = np.array(self.adj_mat.sum(1))
        d_sqrt = np.power(diag, -0.5, dtype=np.float32).squeeze()
        d_sqrt[np.isinf(d_sqrt)] = 0.
        d_mat_inv = sp.diags(d_sqrt)
        self.laplacian_mat = d_mat_inv.dot(self.adj_mat).dot(d_mat_inv)

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.laplacian_mat)
        self.eye_mat = self._convert_sp_mat_to_sp_tensor(sp.eye(self.sparse_norm_adj.shape[0]))
        return self.sparse_norm_adj, self.eye_mat

    def _convert_sp_mat_to_sp_tensor(self, matrix_sp):
        coo = matrix_sp.tocoo()
        idxs = t.LongTensor(np.mat([coo.row, coo.col]))
        vals = t.from_numpy(coo.data.astype(np.float32))  # as_tensor보다 from_numpy가 빠름
        return t.sparse.FloatTensor(idxs, vals, coo.shape)


"""
class Matrix1(object):
    
    Manage all operations according to Matrix creation
    

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_user = df['userId'].max()
        self.n_item = df['movieId'].max()

        self.R = self.create_R_mat()
        self.adj_mat = self.create_adj_mat()
        self.laplacian_mat = self.create_laplacian_mat()
        self.sparse_norm_adj = self.create_sparse_lap_mat()
        self.eye_mat = self._convert_sp_mat_to_sp_tensor(sp.eye(self.sparse_norm_adj.shape[0]))

    def create_R_mat(self):
        R_temp = sp.dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        u_m_set = tuple(zip(self.df['userId'], self.df['movieId']))
        for u, m in u_m_set:
            R_temp[u-1, m-1] = 1.
        return R_temp

    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_user, self.n_user:] = R
        adj_mat[self.n_user:, :self.n_user] = R.T
        return adj_mat.todok()

    def create_laplacian_mat(self):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(self.adj_mat.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5, dtype=np.float32).squeeze()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        lap_mat = d_mat_inv_sqrt.dot(self.adj_mat).dot(d_mat_inv_sqrt)
        return lap_mat.tocoo()

    def _convert_sp_mat_to_sp_tensor(self, matrix_sp):
        coo = matrix_sp.tocoo()
        idxs = t.LongTensor(np.mat([coo.row, coo.col]))
        vals = t.from_numpy(coo.data.astype(t.float32))  # as_tensor보다 from_numpy가 빠름
        return t.sparse.FloatTensor(idxs, vals, coo.shape)

    def create_sparse_lap_mat(self):
        return self._convert_sp_mat_to_sp_tensor(self.laplacian_mat)
"""