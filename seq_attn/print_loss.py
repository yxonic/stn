import sys
import re
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

epoch_pat = re.compile(r'\[([0-9]+):')
loss_pat = re.compile('loss ([0-9.]*)')


def get_loss(logs, from_epoch=0, to_epoch=30):
    last_epoch = from_epoch
    buf = 0
    epochs = []
    losses = []
    for line in logs:
        eo = epoch_pat.search(line)
        lo = loss_pat.search(line)
        if eo is None or lo is None:
            continue
        epoch = int(eo.group(1))
        loss = float(lo.group(1))
        if epoch == last_epoch:
            buf = loss
        elif epoch < last_epoch:
            epochs = []
            losses = []
            last_epoch = epoch
        else:
            epochs.append(last_epoch)
            losses.append(buf)
            last_epoch = epoch
    epochs.append(last_epoch)
    losses.append(buf)
    last_epoch = epoch
    end = to_epoch - from_epoch + 1
    return epochs[:end], losses[:end]


if __name__ == '__main__':
    plt.figure(figsize=(3, 3))
    for i in range(len(sys.argv[1:]) // 2):
        name = sys.argv[i * 2 + 1]
        file = sys.argv[i * 2 + 2]
        x, y = get_loss(list(open(file)), 0, 50)
        plt.plot(x, y, label=name, antialiased=True, linewidth=0.5)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('loss.pdf')
    plt.show()
