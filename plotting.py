import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


LITTLE_DIST = .01

def plot_overlap(states, patterns, file_name='overlap'):
    ax = plt.figure(figsize=(6,3)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(left=-LITTLE_DIST, right=len(states)-1+LITTLE_DIST)
    plt.ylim(bottom=-LITTLE_DIST, top=1+LITTLE_DIST)
    for p in patterns:
        overlap = list(map(lambda s: np.sum(p == s) / p.shape[0], states))
        plt.plot(overlap)
    plt.legend(['pattern_{}'.format(i) for i in range(len(patterns))])
    plt.tight_layout()
    plt.savefig('vis/{}.png'.format(file_name))
    plt.close()

def plot_energy(values, labels, file_name='energy'):
    ax = plt.figure(figsize=(6,4.8)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(left=-LITTLE_DIST, right=max(len(v) for v in values)-1+LITTLE_DIST)
    for v in values:
        plt.plot(v)
    plt.legend(labels)
    plt.tight_layout()
    plt.savefig('vis/{}.png'.format(file_name))
    plt.close()

def plot_bar(value_counts, file_name='counts'):
    plt.figure(figsize=(6,4.8))
    total = np.sum(value_counts)
    ax = value_counts.plot(kind='bar', title='{} runs'.format(total))
    # https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots
    for p in ax.patches:
        text = str(p.get_height())
        text += ' - {:6.2%}'.format(int(text) / total)
        ax.annotate(text, (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('vis/{}.png'.format(file_name))
    plt.close()

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, 2017-2018

width  = None
height = None

def util_setup(w, h):
    global width, height
    width  = w
    height = h

def plot_states(S, file_name='states', titles=None):
    n_cols = np.min([len(S), 5])
    n_rows = int(np.ceil(len(S) / n_cols))
    plt.figure(figsize=(6,n_rows))
    for i, s in enumerate(S):
        title = None if titles is None else titles[i]
        plt.subplot(n_rows, n_cols, i+1, title=title)
        plt.imshow(s.reshape((height, width)), cmap='gray', interpolation='nearest', vmin=-1, vmax=+1)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig('vis/{}.png'.format(file_name))
    plt.close()
