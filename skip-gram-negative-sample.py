# _*_ coding:utf-8 _*_
import numpy as np
import torch
from torch import nn
from collections import Counter
from torch.utils.data import DataLoader

vocab_max_size = 30000
layer1_size = 300
window = 10
b_size = 240
num_epochs = 5
learn_rate = 0.0001
# text_size = 24000000
text_size = 2400
K = 100


class SkipGramModel(nn.Module):

    def __init__(self, V, n):
        super().__init__()
        # V是词汇表大小，n是词向量的维度 V U 分别生成输入和输出词的词向量
        initrange = 0.5 / n
        self.V = nn.Embedding(V, n)
        # 自动把整数转换成onehot然后生成词向量。eg 0-> [1,0] -> ...
        self.V.weight.data.normal_(-initrange, initrange)
        self.U = nn.Embedding(V, n)
        self.U.weight.data.normal_(-initrange, initrange)

    def forward(self, x, positive_w, negative_w):  # contexts, batch64窗口2的话 contexts 64*4， onehot 64*4*V. x 64, one hot 64*V

        vc = self.V(x)  # (64*V)*(V*n) -> 64*n
        u_context = self.U(positive_w)  # (64*4*V) * (V*n)  -> 64*4*n
        u_negative = self.U(negative_w)
        uv_context = torch.bmm(u_context, vc.unsqueeze(2)).squeeze()  # (64,4,1) squeeze 64,4
        uv_negative = torch.bmm(u_negative,-vc.unsqueeze(2)).squeeze()
        log_pos = nn.functional.logsigmoid(uv_context).sum(1)  # batch_size
        log_neg = nn.functional.logsigmoid(uv_negative).sum(1)
        loss = log_pos + log_neg

        if torch.isnan(nn.functional.logsigmoid(uv_context)).sum() > 0:
            print('nan')
        if torch.isinf(nn.functional.logsigmoid(uv_context)).sum() > 0:
            print('inf')
        if torch.isnan(nn.functional.logsigmoid(uv_negative)).sum() > 0:
            print('nan')
        if torch.isinf(nn.functional.logsigmoid(uv_negative)).sum() > 0:
            print('inf')
        if torch.isnan(uv_negative.sum(1)).sum() > 0:
            print('nan')
        if torch.isinf(uv_negative.sum(1)).sum() > 0:
            print('inf')
        return -loss


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, text, word2index, index2word, window_size, word_cnt):
        # index2word是个列表，w2i是个字典
        self.text_index = torch.LongTensor(
            [word2index.get(w, len(index2word) - 1) for w in text])  # 不在字典的词返回unk的index 就是最后一个索引
        self.word2index = word2index
        self.index2word = index2word
        self.window = window_size
        self.word_cnt = torch.Tensor(word_cnt)
        sample_p = (word_cnt / np.sum(word_cnt)) ** (3. / 4.)
        self.negative_sample_p = torch.Tensor(sample_p/np.sum(sample_p))

    def __len__(self):
        return len(self.text_index)

    def __getitem__(self, idx):
        # 返回一个中心词 和他的一堆上下文词
        center_w = self.text_index[idx]
        positive_index = [(idx + i) % len(self.text_index) for i in range(-window, window + 1) if i != 0]
        # % 处理一下 list index out of range
        positive_w = self.text_index[positive_index]
        neg_words = torch.multinomial(self.negative_sample_p, K * positive_w.shape[0], True)
        return center_w, positive_w, neg_words

    def num_words(self):
        return len(self.index2word)


def build_word_dataset(window_size=2):
    news = 0

    if news:
        f = open("data/news.2013.mini.clean3", "r", encoding='UTF-8')
        text = f.read()  # 大概30M
        f.close()


    else:
        f = open("data/text8/text8", "r")
        text = f.read()
        f.close()

        # text = open('data/original_rt_snippets.txt', 'r', encoding='UTF-8').read().replace('\n', ' ')
    # 构造一个字典
    text = text.lower().split()
    text = text[:text_size]
    vocab_count = dict(Counter(text).most_common(vocab_max_size - 1))
    vocab_count['<unk>'] = len(text) - np.sum(list(vocab_count.values()))

    index2word = [w for w in vocab_count.keys()]
    word2index = {w: i for i, w in enumerate(index2word)}

    # vocab_count的频数取出来 不要字符串的key，给负采样用
    word_counts = np.array([cnt for cnt in vocab_count.values()], dtype=np.float32)

    return SkipGramDataset(text, word2index, index2word, window_size, word_counts)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    dataset = build_word_dataset(window)
    # dataset[5]
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True,
                            num_workers=4)
    # num_workers 参数不能用 多线程跑不起？ 解决 是因为要放在main函数。因为多线程程序要放在主函数中训练。
    m = SkipGramModel(dataset.num_words(), layer1_size).to(device)
    print(m)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=learn_rate)
    # 训练
    size = len(dataset)
    m.train()
    all_loss = []
    epochs = num_epochs
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        for batch, (center, positive_w, negative_w) in enumerate(dataloader):
            center, positive_w, negative_w = center.type(torch.LongTensor).to(device), positive_w.type(torch.LongTensor).to(device), negative_w.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            loss = m(center, positive_w, negative_w).mean()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                all_loss.append(loss.item())
                loss, current = loss.item(), batch * len(center)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # if t > 0:
        #     f = open("save-news-24m/loss-epoch-" + str(t) + ".txt", 'w')
        #     for i in all_loss:
        #         f.write(str(i) + '\n')
        #     f.close()
        #     torch.save(m, "save-news-24m/model-epoch-" + str(t) + ".pth")

    torch.save(m, 'model.pth')

    f = open("loss.txt", 'w')
    for i in all_loss:
        f.write(str(i) + '\n')
    f.close()
    print('hi')
