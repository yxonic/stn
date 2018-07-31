"""Plot accuracy bar chart. (deprecated) """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

N = 4
model_name = {
    'none': 'Enc-Dec',
    'dot': 'Attn-Dot',
    'fc': 'Attn-FC',
    'fc_pos': 'Attn-Pos',
    'stnm': 'STNM',
    'stnr': 'STNR'
}

# model_color = {
#     'none': '#9467bd',
#     'dot': '#8c564b',
#     'fc': '#1f77b4',
#     'fc_pos': '#2ca02c',
#     'stnm': '#ff7f0e',
#     'stnr': '#d62728'
# }

model_color = {
    'none': '#8c564b',
    'dot': '#9467bd',
    'fc': '#d62728',
    'fc_pos': '#2ca02c',
    'stnm': '#ff7f0e',
    'stnr': '#1f77b4'
}

ind = np.arange(N)  # the x locations for the groups
width = 0.15    # the width of the bars

fig = plt.figure(figsize=(12, 2.5))


gs = gridspec.GridSpec(1, 3)
ax = [plt.subplot(gs[x]) for x in range(3)]


accs = {
    'none': (0.266029, 0.271672, 0.277018, 0.28224),
    'dot': (0.523738, 0.547562, 0.580187, 0.616718),
    'fc': (0.682897, 0.709841, 0.72982, 0.755914),
    'fc_pos': (0.725089, 0.735569, 0.740864, 0.758405),
    'stnm': (0.7295361, 0.733221, 0.749076, 0.758764),
    'stnr': (0.739318, 0.74796, 0.758483, 0.766966)
}

w = 0
for i, name in enumerate(['none', 'dot', 'fc', 'fc_pos',
                          'stnm', 'stnr']):
    ax[0].bar(ind + w, accs[name], width - 0.06, color=model_color[name],
              edgecolor='black', linewidth=1.5, align='edge')
    w += width

# add some text for labels, title and axes ticks
ax[0].set_ylabel('Accuracy')
ax[0].set_title('(a) Melody')
ax[0].set_xticks(ind + width * 2.5)
ax[0].set_ylim(0.2, 0.8)
ax[0].set_yticks([0.2, 0.4, 0.6, 0.8])
ax[0].set_xticklabels(('60\\%', '70\\%', '80\\%', '90\\%'))


accs = {
    'none': (0.404654, 0.427566, 0.444819, 0.450691),
    'dot': (0.530218, 0.562676, 0.599817, 0.611405),
    'fc': (0.656929, 0.700796, 0.717454, 0.725075),
    'fc_pos': (0.715528, 0.723755, 0.732313, 0.740828),
    'stnm': (0.716561, 0.726172, 0.740232, 0.748681),
    'stnr': (0.739286, 0.750979, 0.759111, 0.777183)
}

w = 0
for i, name in enumerate(['none', 'dot', 'fc', 'fc_pos',
                          'stnm', 'stnr']):
    ax[1].bar(ind + w, accs[name], width - 0.06, color=model_color[name],
              edgecolor='black', linewidth=1.5, align='edge')
    w += width

# add some text for labels, title and axes ticks
ax[1].set_ylabel('Accuracy')
ax[1].set_title('(b) Formula')
ax[1].set_ylim(0.2, 0.8)
ax[1].set_yticks([0.2, 0.4, 0.6, 0.8])
ax[1].set_xticks(ind + width * 2.5)
ax[1].set_xticklabels(('60\\%', '70\\%', '80\\%', '90\\%'))


accs = {
    'none': (0.217697, 0.226864, 0.251381, 0.267301),
    'dot': (0.333594, 0.446573, 0.554849, 0.599926),
    'fc': (0.6144, 0.641969, 0.686126, 0.707026),
    'fc_pos': (0.623731, 0.65243, 0.697736, 0.719943),
    'stnm': (0.673731, 0.705349, 0.730993, 0.734305),
    'stnr': (0.711766, 0.735683, 0.754653, 0.760411)
}

w = 0
for i, name in enumerate(['none', 'dot', 'fc', 'fc_pos',
                          'stnm', 'stnr']):
    ax[2].bar(ind + w, accs[name], width - 0.06, color=model_color[name],
              edgecolor='black', linewidth=1.5, align='edge')
    w += width

# add some text for labels, title and axes ticks
ax[2].set_ylabel('Accuracy')
ax[2].set_title('(c) Multi-Line')
ax[2].set_ylim(0.2, 0.8)
ax[2].set_yticks([0.2, 0.4, 0.6, 0.8])
ax[2].set_xticks(ind + width * 2.5)
ax[2].set_xticklabels(('60\\%', '70\\%', '80\\%', '90\\%'))


plt.tight_layout()
plt.savefig('acc.pdf')
