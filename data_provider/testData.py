from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_hour

if __name__ == '__main__':

    data_set = Dataset_ETT_hour(
        root_path='../datasets/ETT-small/',
        data_path='ETTh1.csv',
        flag='train',
        features='M',
        size=[96, 0, 96],
        target='OT',
        percent=100
    )
    data_loader = DataLoader(
        data_set,
        batch_size=256,
        shuffle=True,
        drop_last=True)
    print(data_set.__getitem__(11111111111111))
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        print(batch_x.shape)