import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import MovieLens
from utils import Download
from matrix import Matrix
from model import NGCF
from bprloss import BPR
from experiment import Train, Test
from parsers_ngcf import args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

root_path = '../dataset'
dataset = Download(root=root_path, file_size=args.file_size, download=False)
total_df, train_df, test_df = dataset.split_train_test()
n_user = total_df['userId'].max()
n_item = total_df['movieId'].max()

train_set = MovieLens(train_df, total_df, train=True, ng_ratio=1)
test_set = MovieLens(test_df, total_df, train=False, ng_ratio=99)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

sparse_lap_mat, eye_mat = Matrix(df=total_df, device=device).create_matrix()

model = NGCF(n_user=n_user,
             n_item=n_item,
             embed_size=64,
             layer_size=[64, 64, 64],
             node_dropout=0.2,
             mess_dropout=[0.1, 0.1, 0.1],
             lap_mat=sparse_lap_mat,
             eye_mat=eye_mat,
             device=device).to(device=device)

if __name__ == '__main__':

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = BPR(weight_decay=0.025, batch_size=args.batch_size)

    train = Train(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  train_dataloader=train_loader,
                  test_dataloader=test_loader,
                  epochs=args.epoch,
                  device=device)
    train.train()
    print('train ended')

    test = Test(model=model,
                dataloader=test_loader,
                ks=args.ks,
                device=device)









