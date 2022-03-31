# _*_ coding:utf-8 _*_
import numpy as np
import torch
from torch import nn
from collections import Counter
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

vocab_max_size = 30000
layer1_size = 300
window = 10
b_size = 240
num_epochs = 5
learn_rate = 0.0001
text_size = 24000000


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

    def forward(self, x, contexts):  # contexts, batch64窗口2的话 contexts 64*4， onehot 64*4*V. x 64, one hot 64*V
        vc = self.V(x)  # (64*V)*(V*n) -> 64*n
        u_context = self.U(contexts)  # (64*4*V) * (V*n)  -> 64*4*n
        uv_context = torch.bmm(u_context, vc.unsqueeze(2)).squeeze()  # (64,4,1) squeeze 64,4
        u_all = vc @ self.U.weight.data.T  # (64,V)
        # loss = - uv_context.sum(1) + 2 * window * torch.log(torch.exp(u_all).sum(1))  # inf
        loss = 2 * window * torch.logsumexp(u_all, 1) - uv_context.sum(1)

        if torch.isnan(torch.logsumexp(u_all, 1)).sum() > 0:
            print('nan')
        if torch.isinf(torch.logsumexp(u_all, 1)).sum() > 0:
            print('inf')
        if torch.isnan(uv_context.sum(1)).sum() > 0:
            print('nan')
        if torch.isinf(uv_context.sum(1)).sum() > 0:
            print('inf')

        return loss


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, text, word2index, index2word, window):
        # index2word是个列表，w2i是个字典
        self.text_index = torch.LongTensor(
            [word2index.get(w, len(index2word) - 1) for w in text])  # 不在字典的词返回unk的index 就是最后一个索引
        self.word2index = word2index
        self.index2word = index2word
        self.window = window

    def __len__(self):
        return len(self.text_index)

    def __getitem__(self, idx):
        # 返回一个中心词 和他的一堆上下文词
        center_w = self.text_index[idx]
        context_pos = [(idx + i) % len(self.text_index) for i in range(-window, window + 1) if i != 0]
        # % 处理一下 list index out of range
        return center_w, self.text_index[context_pos]

    def num_words(self):
        return len(self.index2word)


def build_word_dataset(window_size=2):
    news = 1

    if news:
        text = open("data/news.2013.mini.clean3", "r", encoding='UTF-8').read()  # 大概30M


    else:
        text = open("data/text8/text8", "r").read()

        text = open('data/original_rt_snippets.txt', 'r', encoding='UTF-8').read().replace('\n', ' ')
    # 构造一个字典
    text = text.lower().split()
    text = text[:text_size]
    vocab_count = dict(Counter(text).most_common(vocab_max_size - 1))
    vocab_count['<unk>'] = len(text) - np.sum(list(vocab_count.values()))

    index2word = [w for w in vocab_count.keys()]
    word2index = {w: i for i, w in enumerate(index2word)}

    return SkipGramDataset(text, word2index, index2word, window_size)


if __name__ == '__main__':

    dataset = build_word_dataset(window)

    m = torch.load('save/save-news-24m/model-epoch-1.pth')

    V = m.V.weight.data.detach().cpu()
    U = m.U.weight.data.detach().cpu()
    embedding_matrix = U + V
    embedding_matrix = nn.functional.normalize(embedding_matrix).numpy()
    org_em = (U + V).numpy()

    pca = PCA(2)
    words = ['school', 'university', 'teacher', 'professor',
             'grade', 'paper', 'class', 'lesson', 'exam', 'prom',
             'academy', 'education', 'college', 'exams', 'gpa',
             'food', 'spaghetti', 'beer', 'wine', 'burger', 'steak',
             'kfc', 'meat', 'chicken', 'pork', 'beef']
    w_emb, words_found = [], []

    for word in words:
        if word in dataset.word2index.keys():
            emb = embedding_matrix[dataset.word2index[word]]
            w_emb.append(emb)
            words_found.append(word)
    w_emb = np.vstack(w_emb)

    embedding_pca = pca.fit_transform(w_emb)

    plt.figure(figsize=(10, 5))
    for word, embedding in zip(words_found, embedding_pca):
        plt.scatter(embedding[0], embedding[1], marker='x', color='red')
        plt.text(embedding[0] + 0.001, embedding[1] + 0.001, word, fontsize=9)
    plt.show()

    results = {}

    with open('data/questions-words.txt', 'r') as f:
        lines = f.read().split('\n')

    i = 0

    for line in lines:
        i += 1
        analogy = line.split()
        if not analogy:
            continue

        if ':' in line:
            curr_topic = analogy[-1]
            results[curr_topic] = {'total': 0, 'correct': 0}
            continue

        if any(word.lower() not in dataset.word2index for word in analogy):
            continue

        results[curr_topic]['total'] += 1
        analogy_index = [dataset.word2index[w.lower()] for w in analogy]
        word1, word2, word3, word4 = analogy_index

        candidate = (embedding_matrix[word2] - embedding_matrix[word1] + embedding_matrix[word3]).reshape(1, -1)

        # closest_index = np.argsort(cdist(candidate, org_em, metric='cosine').reshape(-1))[:10]  #因为刚刚有L2norm 就是cos距离分母上面这一项
        dist = candidate @ embedding_matrix.T
        closest_index = np.argsort(-dist.reshape(-1))[:5]

        for i in closest_index:
            if i in [word1, word2, word3]:
                continue
            elif i == word4:
                results[curr_topic]['correct'] += 1
    all_question = 0
    y = 0
    for key, value in results.items():
        print(key, "accuracy: ", "%.2f" % (value['correct'] / value['total'] * 100), "%", "(", value['correct'], ',',
              value['total'], ")")
        all_question += value['total']
        y += value['correct']

    print("all accuracy: ", "%.2f" % (y / all_question * 100), "%", "(", y, ',',
          all_question, ")")
    print('hi')
