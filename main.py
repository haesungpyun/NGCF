import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import numpy as np

from utils import MovieLens
from utils import Download
from matrix import Matrix
from model import NGCF
from bprloss import BPR
from experiment import Train, Test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

root_path = 'dataset'
dataset = Download(root=root_path, file_size='100k', download=False)
total_df, train_df, test_df = dataset.split_train_test()
n_user = total_df['userId'].max()
n_item = total_df['movieId'].max()

train_set = MovieLens(train_df, total_df, train=True)
test_set = MovieLens(test_df, total_df, train=False)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

for u, p, n in train_loader:
    print(u.shape)
    print(p.shape)
    print(n.shape)
    break
for u, p in test_loader:
    print(u.shape)
    print(p.shape)
    break


sparse_lap_mat, eye_mat = Matrix(total_df).create_matrix()

if __name__ == '__main__':
    model = NGCF(n_user=n_user,
                 n_item=n_item,
                 embed_size=64,
                 layer_size=[64, 64, 64],
                 node_dropout=0.2,
                 mess_dropout=[0.1, 0.1, 0.1],
                 lap_mat=sparse_lap_mat,
                 eye_mat=eye_mat,
                 device='cpu').to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = BPR(weight_decay=0.025, batch_size=256)

    test = Test(model=model,
                dataframe=test_df,
                dataloader=test_loader,
                epochs=1,
                ks=10,
                device='cpu')
    test.eval()

    train = Train(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  dataloader=train_loader,
                  epochs=1,
                  device='cpu').train()
    print('train ended')


