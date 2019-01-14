import time

import numpy as np
import pandas as pd

from hopfield import *
from plotting import *

randomState = np.random.RandomState(seed=42)

### load data

patterns = []
with open('./data/letters.txt') as f:
    count = int(f.readline())
    for _ in range(count):
        height, width = map(int, f.readline().split())
        pattern = np.empty((height, width), dtype=int)
        for row in range(height):
            pattern[row,:] = list(map(int, f.readline().strip().split()))
        patterns.append(pattern.flatten())

util_setup(width, height)

### training

model = Hopfield(np.prod(patterns[0].shape))
model.train(patterns)

### corrupted patterns

E_values = []
E_labels = []
for k in [7*i for i in range(4)]:
    corrupted_patterns = np.copy(patterns)
    for p_i,x in enumerate(corrupted_patterns):
        if k > 0:
            for corruptedIdx in randomState.randint(x.shape[0], size=k):
                x[corruptedIdx] *= -1

        S, E = model.run_sync(x)
        E_values.append(E)
        E_labels.append('pattern{}k{}'.format(p_i, k))

        plot_states(S, file_name='corrupted_{}_k_{}_states'.format(p_i, k))
        plot_overlap(S, patterns, file_name='corrupted_{}_k_{}_overlap'.format(p_i, k))

    plot_states(corrupted_patterns, file_name='corrupted_k_{}'.format(k))

plot_energy(E_values, E_labels, file_name='corrupted_energy')
for i,_ in enumerate(patterns):
    values = [v for p_i,v in enumerate(E_values) if p_i % len(patterns) == i]
    labels = [l for p_i,l in enumerate(E_labels) if p_i % len(patterns) == i]
    plot_energy(values, labels, file_name='corrupted_{}_energy'.format(i))

### random inputs

for r_i in range(5):
    x = randomState.choice([-1,1], size=len(patterns[0]))
    S, E = model.run_sync(x)
    plot_states(S, file_name='random_{}_states'.format(r_i))
    plot_overlap(S, patterns, file_name='random_{}_overlap'.format(r_i))
    plot_energy([E], [r_i], file_name='random_{}_energy'.format(r_i))

### dynamics

true_attractors = [encode_state(p) for p in patterns]

#for sample_size in [5000, 10000, 20000, 50000]:
for sample_size in [5000]:
    final_states = pd.DataFrame(columns=['count'])
    counter_true, counter_spurious, counter_cycle = 0, 0, 0

    hopfield_time = 0.
    # t_start = time.clock()
    for r_i in range(sample_size):
        x = randomState.choice([-1, 1], size=len(patterns[0]))
        start = time.clock()
        s, cycle_detected = model.get_last_state(x)
        hopfield_time += time.clock() - start

        s_str = encode_state(s)
        if cycle_detected:
            counter_cycle += 1
        elif s_str in true_attractors:
            counter_true += 1
        else:
            counter_spurious += 1

        if s_str in final_states.index:
            final_states.at[s_str, 'count'] += 1
        elif s_str[::-1] in final_states.index:
            final_states.at[s_str[::-1], 'count'] += 1
        else:
            temp = encode_state(np.transpose(np.transpose(s.reshape(height, width))[::-1]).flatten())
            if temp in final_states.index:
                final_states.at[temp, 'count'] += 1
            elif temp[::-1] in final_states.index:
                final_states.at[temp[::-1], 'count'] += 1
            else:
                final_states.at[s_str, 'count'] = 1
    # t_end = time.clock()
    # print('loop time ~:     {:6.3} seconds'.format(t_end - t_start))
    print('hopfield time ~: {:6.3} seconds'.format(hopfield_time))
    print('(this is CPU time only on Unix, NOT on WINDOWS)')

    counts = pd.Series({ 'true attractors':counter_true,
                         'spurious attractors':counter_spurious,
                         'limit cycles':counter_cycle })
    plot_bar(counts, file_name='sample_{}_counts'.format(sample_size))

    top_20 = final_states.sort_values('count', ascending=False).head(20)
    plot_states([np.array(decode_state(p)) for p in top_20.index],
                file_name='sample_{}_top20attractors'.format(sample_size),
                titles=top_20['count'].astype(int))
