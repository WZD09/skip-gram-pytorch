# _*_ coding:utf-8 _*_
import os

import numpy as np
import re
import torch
from torch import nn
from collections import Counter
from torch.utils.data import DataLoader

vocab_max_size = 30000
layer1_size = 100
window = 10
b_size = 240
num_epochs = 30
learn_rate = 0.0001
K = 5


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

    def forward(self, x, positive_w,
                negative_w):  # contexts, batch64窗口2的话 contexts 64*4， onehot 64*4*V. x 64, one hot 64*V

        vc = self.V(x)  # (64*V)*(V*n) -> 64*n
        u_context = self.U(positive_w)  # (64*4*V) * (V*n)  -> 64*4*n
        u_negative = self.U(negative_w)
        uv_context = torch.bmm(u_context, vc.unsqueeze(2)).squeeze()  # (64,4,1) squeeze 64,4
        uv_negative = torch.bmm(u_negative, -vc.unsqueeze(2)).squeeze()
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
    def __init__(self, text, word2index, index2word, window_size, word_cnt, text_id_index):
        # index2word是个列表，w2i是个字典
        self.text_index = torch.LongTensor(
            [word2index.get(w, len(index2word) - 1) for w in text])  # 不在字典的词返回unk的index 就是最后一个索引
        self.word2index = word2index
        self.index2word = index2word
        self.window = window_size
        self.word_cnt = torch.Tensor(word_cnt)
        sample_p = (word_cnt / np.sum(word_cnt)) ** (3. / 4.)
        self.negative_sample_p = torch.Tensor(sample_p / np.sum(sample_p))
        self.text_id_index = text_id_index

    def __len__(self):
        return len(self.text_index)

    def __getitem__(self, idx):
        # 返回一组中心词 和他的一堆上下文词
        center_w = self.text_index[idx]
        positive_index = [(idx + i) % len(self.text_index) for i in range(-window, window + 1) if i != 0]
        positive_index.append(self.text_id_index[idx])
        # % 处理一下 list index out of range
        positive_words = self.text_index[positive_index]
        # 因为是sentence vector所以每个中心词的上下文都要加上句子id的向量跟上下文一起
        neg_words = torch.multinomial(self.negative_sample_p, K * positive_words.shape[0], True)

        return center_w, positive_words, neg_words

    def num_words(self):
        return len(self.index2word)


def _tokenize_str(str_):
    # 只保留字母、数字和标点符号 跟源码一样处理输入
    str_ = re.sub(r'[^A-Za-z0-9(),.!?_*\'`]', ' ', str_)

    str_ = re.sub(r'\s{2,}', ' ', str_)

    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)

    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)

    return str_.strip().lower().split()


def build_word_dataset(window_size=2):

    content = []
    labels = {}
    for location in ['train-pos.txt', 'train-neg.txt', 'test-pos.txt', 'test-neg.txt', 'unsup.txt']:
        file_list = []
        whole_location= 'data/aclImdb/all/'+location
        with open(whole_location, 'r', encoding='utf8') as f:
            line = f.readlines()
            file_list.extend(line)
        labels[location] = len(file_list)
        content.extend(file_list)
    labels['all'] = len(content)
    np.save('imdb-save/labels.npy', labels)
    print(labels)

    # 构造一个字典
    content = ['_*' + str(i) + ' ' + content[i] for i in range(len(content))]

    text = [_tokenize_str(line) for line in content]

    all_text = [y for l in text for y in l]
    text_counter = Counter(all_text)
    min_count_index = list(dict(text_counter.most_common()).values()).index(1)
    vocab_count = dict(text_counter.most_common(min_count_index - 1))
    vocab_count['<unk>'] = len(all_text) - np.sum(list(vocab_count.values()))


    sen_id2index = {}

    for text_i in range(len(text)):
        sentence = text[text_i]
        sen_id = sentence[0]  # '_*0'
        sen_id2index[sen_id] = 1

    vocab_count.update(sen_id2index)

    index2word = [w for w in vocab_count.keys()]
    word2index = {w: i for i, w in enumerate(index2word)}
    np.save('imdb-save/word2index.npy', word2index)
    # vocab_count的频数取出来 不要字符串的key，给负采样用
    word_counts = np.array([cnt for cnt in vocab_count.values()], dtype=np.float32)

    text_id_index = []

    for text_i in range(len(text)):
        sentence = text[text_i]
        sen_length = len(sentence)
        sen_id = sentence[0]  # '_*0'
        sen_id_index = word2index[sen_id]
        text_id_index.append([sen_id_index] * sen_length)
    text_id_index = [y for l in text_id_index for y in l]

    return SkipGramDataset(all_text, word2index, index2word, window_size, word_counts, text_id_index)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    dataset = build_word_dataset(window)
    # dataset[5]
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True,
                            num_workers=4)

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
            center, positive_w, negative_w = center.type(torch.LongTensor).to(device), positive_w.type(
                torch.LongTensor).to(device), negative_w.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            loss = m(center, positive_w, negative_w).mean()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                all_loss.append(loss.item())
                loss, current = loss.item(), batch * len(center)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if t % 8 == 0:
            f = open("imdb-save/loss-epoch-" + str(t) + ".txt", 'w')
            for i in all_loss:
                f.write(str(i) + '\n')
            f.close()
            torch.save(m, "imdb-save/model-epoch-" + str(t) + ".pth")

    torch.save(m, 'imdb-save/model.pth')

    f = open("loss.txt", 'w')
    for i in all_loss:
        f.write(str(i) + '\n')
    f.close()
    print('hi')
