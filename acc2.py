# _*_ coding:utf-8 _*_
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

vocab_max_size = 30000
layer1_size = 300
window = 10
b_size = 240
num_epochs = 5
learn_rate = 0.0001
text_size = 24000000

class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    a = np.loadtxt('imdb-save/wzd/test.out')  # shape 385416 100
    word2index = np.load('imdb-save/wzd/word2index.npy', allow_pickle=True)
    word2index = dict(word2index.item())
    vocab_size = len(word2index)
    a = np.concatenate((a[:vocab_size, :], a[vocab_size:, :]), axis=1)
    sen_id = []
    for word in word2index:
        if word.find('_*') == 0:
            sen_id.append(word2index[word])
    if len(sen_id ) != 100000:
        print('not imdb dataset length')
    sen_vec = a[sen_id,:100]
    pos_label = [1 for i in range(12500)]
    neg_label = [0 for i in range(12500)]
    train_label = np.array(pos_label+neg_label)
    test_label = np.array(pos_label+neg_label)
    train_x = sen_vec[:25000]
    test_x = sen_vec[25000:50000]
    dataset = ImdbDataset(train_x, train_label)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                            num_workers=4)

    net = torch.nn.Sequential(
        torch.nn.Linear(100, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2),
        nn.Softmax(dim=1)
    )
    net=net.double().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_all = []
    for epoch in range(3):
        for batch,(x,y) in enumerate(dataloader):
            out = net(x.to(device))
            loss = loss_func(out, y.to(device))
            loss_all.append(loss.data.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            ty = net(torch.from_numpy(test_x).to(device))
            acc=(torch.sum(torch.max(ty, dim=1)[1]).data.item())/np.sum(test_label)
        print(loss, acc)
    plt.plot(loss_all)