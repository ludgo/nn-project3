import itertools
import functools

import numpy as np


def encode_state(state):
    if state[0] == 1:
        # ensure starts with 0
        # this removes difference between pattern and its negative
        return functools.reduce(lambda a, b: a + b, ['0' if num == 1 else '1' for num in state])
    return functools.reduce(lambda a, b: a + b, ['1' if num == 1 else '0' for num in state])

def decode_state(state_str):
    return [1 if c == '1' else -1 for c in state_str]

class Hopfield():

    def __init__(self, dim):
        self.dim  = dim

    def train(self, patterns):
        self.W = np.zeros((self.dim, self.dim))
        P = len(patterns)
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    for p_i in range(P):
                        self.W[i, j] += patterns[p_i][i] * patterns[p_i][j]
                    self.W[i, j] *= 1./P

    def energy(self, s):
        energy = 0.
        for j in range(self.dim):
            for i in range(self.dim):
                if i != j:
                    energy += self.W[i, j] * s[i] * s[j]
        energy *= -.5
        return energy

    def sign(self, array):
        n_s = len(array)
        s = np.ones(n_s, dtype=int)
        for s_i in range(n_s):
            if array[s_i] < 0:
                s[s_i] = -1
        return s

    def forward(self, s, neuron=None):
        net = np.asarray([net_i @ s for net_i in self.W])
        s = self.sign(net)
        return s if neuron is None else s[neuron]

    def run_sync(self, x, eps=None):
        s = x.copy()
        e = self.energy(s)
        S = [s]
        E = [e]

        for _ in itertools.count() if eps is None else range(eps):

            s = self.forward(s, neuron=None)
            e = self.energy(s)

            S.append(s)
            E.append(e)

            seq_i = len(S)-1
            seq_size = 1
            while seq_size <= len(S) // 2:
                if np.array_equal(S[seq_i - seq_size : seq_i], S[seq_i:]):
                    return S, E
                seq_i -= seq_size
                seq_size *= 2

        return S, E

    def get_last_state(self, x, eps=20):
        s = x.copy()
        S = [s]

        for _ in range(eps):
            s = self.forward(s, neuron=None)
            S.append(s)

            seq_i = len(S)-1
            seq_size = 1
            while seq_size <= len(S) // 2:
                if np.array_equal(S[seq_i - seq_size : seq_i], S[seq_i:]):
                    assert(seq_size <= 2), 'Detected cycle with size > 2.'
                    return S[-1], seq_size != 1
                seq_i -= seq_size
                seq_size *= 2

        assert(False), 'All epochs finished.'
        return S[-1], False
