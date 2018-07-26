import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

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

# model_color = {
#     'none': '#cc6600',
#     'dot': '#cc66ff',
#     'fc': '#3399ff',
#     'fc_pos': '#33cc33',
#     'stnm': '#ff9933',
#     'stnr': '#ff6666'
# }

model_color = {
    'none': '#8c564b',
    'dot': '#9467bd',
    'fc': '#d62728',
    'fc_pos': '#2ca02c',
    'stnm': '#ff7f0e',
    'stnr': '#1f77b4'
}
patches = []
for model in model_color:
    patches.append(mpatches.Patch(color=model_color[model],
                                  label=model_name[model],
                                  linewidth=1))

fig = plt.figure(figsize=(7, 0.4))

fig.legend(handles=patches, ncol=6)
plt.savefig('legend.pdf')
