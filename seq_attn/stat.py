import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from matplotlib import rc
import matplotlib.ticker as ticker
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


def _main(name, ax, bins, max, color, caption, func):
    df = pd.read_table('data/%s_label.txt' % name, converters={
        'label': func
    })
    print(df['label'].sum())
    print((df['label'] > 10).sum() / len(df['label']))
    ax.hist(df['label'], bins=bins, range=(1, max), color=color,
            alpha=0.8, normed=True)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0, xmax=1,
                                                         symbol=''))
    ax.set_ylabel('Percentage (\\%)')
    ax.set_xlabel('Content length')
    ax.set_title(caption, y=-0.4)


if __name__ == '__main__':
    fig = plt.figure(figsize=(13, 3))
    gs = gridspec.GridSpec(1, 5)
    ax = [plt.subplot(gs[x]) for x in range(5)]
    f = lambda x: len(x.split(' '))
    _main('melody', ax[0], 49, 50, 'b', '(a) Melody', f)
    _main('formula', ax[1], 49, 50, 'b', '(b) Formula', f)
    _main('multiline', ax[2], 50, 100, 'b', '(c) Multi-Line', f)
    _main('svt', ax[3], 49, 50, 'r', '(d) SVT', len)
    _main('iiit5k', ax[4], 49, 50, 'r', '(e) IIIT5K', len)
    plt.tight_layout()
    fig.savefig('stat.pdf')
