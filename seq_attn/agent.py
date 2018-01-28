import torch
import torch.nn as nn

from .util import var


class MarkovPolicy(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.state_size = state_size
        self.pi = nn.Linear(state_size, 3)
        self.q = nn.Linear(state_size + 3, 1)

    def forward(self, s_, state, h_):
        return self.pi(state), None

    def value(self, state, action, h_):
        return self.q(torch.cat([state, action], dim=1))

    def default_h(self):
        return None


class RNNPolicy(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(3, hidden_size, 1)
        self.pi = nn.Linear(state_size + hidden_size, 3)
        self.value = nn.Linear(state_size + hidden_size + 3, 1)

    def forward(self, s_, state, h_):
        _, h = self.rnn(s_.view(1, 1, 3), h_)
        return self.pi(torch.cat([state, h.view(1, -1)], dim=1)), h

    def value(self, state, action, h):
        return self.q(torch.cat([state, h.view(1, -1), action], dim=1))

    def default_h(self):
        return var(torch.zeros(1, 1, self.hidden_size))
