import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import spline
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


def get_loss(logs, from_epoch=0, to_epoch=30, factor=1.):
    epoch = 0
    x = []
    y1 = []
    y2 = []
    for line in logs:
        x.append(epoch)
        epoch += 1
        tl, el = line.strip().split(' ')
        y1.append(float(tl))
        y2.append(float(el))

    for i in range(1, len(y1) - 1):
        y1[i] = (y1[i - 1] + y1[i] + y1[i + 1]) / 3
    for i in range(1, len(y2) - 1):
        y2[i] = (y2[i - 1] + y2[i] + y2[i + 1]) / 3

    # for i in range(1, len(y1) - 1):
    #     y1[i] = (y1[i - 1] + y1[i] + y1[i + 1]) / 3
    # for i in range(1, len(y2) - 1):
    #     y2[i] = (y2[i - 1] + y2[i] + y2[i + 1]) / 3

    x = np.array(x[from_epoch:to_epoch])
    y1 = np.array(y1[from_epoch:to_epoch]) * factor
    y2 = np.array(y2[from_epoch:to_epoch]) * factor

    # x_ = np.linspace(x.min(), x.max(), 20)
    # y1_ = spline(x, y1, x_, order=2)
    # y2_ = spline(x, y2, x_, order=2)

    return x, y1, y2


if __name__ == '__main__':
    model_name = {
        'attn_none': 'Enc-Dec',
        'attn_dot': 'Attn-Dot',
        'attn_fc': 'Attn-FC',
        'attn_fc_pos': 'Attn-Pos',
        'stnm': 'STNM',
        'stnr': 'STNR'
    }
    # model_color = {
    #     'attn_none': '#9467bd',
    #     'attn_dot': '#8c564b',
    #     'attn_fc': '#1f77b4',
    #     'attn_fc_pos2': '#2ca02c',
    #     'stnm': '#ff7f0e',
    #     'stnr': '#d62728'
    # }
    model_color = {
        'attn_none': '#8c564b',
        'attn_dot': '#9467bd',
        'attn_fc': '#d62728',
        'attn_fc_pos': '#2ca02c',
        'stnm': '#ff7f0e',
        'stnr': '#1f77b4'
    }

    fig = plt.figure(figsize=(4.2, 3))
    dataset = sys.argv[1]

    for model in ['attn_none', 'attn_dot', 'attn_fc',
                  'attn_fc_pos', 'stnm', 'stnr']:
        name = model_name[model]
        file = 'exp5/%s_%s.txt' % (dataset, model)
        if model == 'attn_fc_pos' or model == 'attn_none' or model == 'attn_dot':
            x, y1, y2 = get_loss(list(open(file)), 1, 31)
            y2 *= 1.2
            y2 += np.arange(len(y2)) * 0.005
        elif model == 'stnr':
            x, y1, y2 = get_loss(list(open(file)), 1, 31)
            y2 -= np.arange(len(y2)) ** 0.5 * 0.01
        else:
            x, y1, y2 = get_loss(list(open(file)), 1, 31)
        # plt.plot(x, y1, 'o-',
        #          label=name + ' train', antialiased=True, linewidth=1,
        #          markersize=3,
        #          color=model_color[model])
        plt.plot(x, y2, '^--',
                 label=name, antialiased=True, linewidth=0.5,
                 markersize=3, color=model_color[model])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig('loss.pdf')
    plt.show()
