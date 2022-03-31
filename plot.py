import matplotlib.pyplot as plt

if __name__ == '__main__':
    loss = []
    with open('save/save-news-24m/loss-epoch-1.txt', 'r') as f:
        for line in (f.read().split('\n')):
            if line:
                loss.append(float(line))
    print(len(loss))
    y = loss
    plt.plot(list(range(len(y))), y, 'r-')
    plt.show()
