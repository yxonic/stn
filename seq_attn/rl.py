import torch
import logging
import torch.nn.functional as F
import numpy as np
import random

from yata.fields import Categorical

from .util import var, save_snapshot, progress, normalize


class ActorCritic:
    def __init__(self, model, seq_model, gamma=0.99, alpha=10.):
        self.model = model
        self.seq_model = seq_model
        self.gamma = gamma
        self.alpha = 10
        self.exp_rate = 0.

    @progress(print_every=32, logger=logging.getLogger())
    def reinforce(self, data, chars, only_value=False):
        if only_value:
            params = self.model.value.parameters()
        else:
            params = filter(lambda p: p.requires_grad,
                            self.model.parameters())
        optimizer = torch.optim.Adam(params)

        yield len(data.keys)
        N = 0
        saved = []
        rewards = []

        for key, item in data.shuffle().epoch(1, backend='torch'):
            L = item.y.size(1) * 2 + 20
            h0 = self.model.get_initial_state(var(item.file))
            for K in range(8):
                N += 1
                # explore
                h = h0
                x = var(torch.zeros(1, 1).long() + 1)
                h_seq = self.seq_model.default_h()

                ss = []
                y_pred = [0] * L
                for i in range(L):
                    y, v, h_ = self.model(x, h)
                    # y.data.clamp_(-5., 5.)
                    # v.data.clamp_(-1., 1.)

                    y_s, h_seq_ = self.seq_model(x, h_seq)

                    if random.random() < self.exp_rate:
                        y = y_s
                    # else:
                    #     y += y_s

                    probs = F.softmax(y.squeeze(), dim=0)
                    dist = Categorical(probs)
                    x_ = dist.sample()
                    ss.append((dist.log_prob(x_), v.squeeze()))
                    x_ = x_.data[0]
                    y_pred[i] = x_
                    if x_ == 0 or i == L - 1:
                        break
                    x = var(torch.LongTensor([[x_]]))
                    h = h_  # var(h_.data)
                    h_seq = h_seq_

                if i == 0:
                    latex = ''
                else:
                    # latex = ''.join(chars.get_original(y_pred[:i]))
                    ws = chars.get_original(item.y.squeeze())
                    tlatex = ws[0]
                    w_ = ws[0]
                    for w in ws[1:]:
                        if w_.startswith('\\') and w.isalnum():
                            tlatex += ' ' + w
                        else:
                            tlatex += w
                        w_ = w

                    ws = chars.get_original(y_pred[:i])
                    latex = ws[0]
                    w_ = ws[0]
                    for w in ws[1:]:
                        if w_.startswith('\\') and w.isalnum():
                            latex += ' ' + w
                        else:
                            latex += w
                        w_ = w

                r = get_return_pixel(latex, item.file)
                yield [('return', r)]
                R = r
                rs = []
                for i in range(len(ss)):
                    rs.insert(0, R)
                    R = self.gamma * R

                saved.extend(ss)
                rewards.extend(rs)

                if N % 3 == 0:
                    print(tlatex, '\t<==>\t', latex, ':', r)

            if N % 32 == 0 or N == len(data.keys):
                rewards = normalize(torch.Tensor(rewards))

                p_loss = 0.
                v_loss = 0.
                for (log, v), reward in zip(saved, rewards):
                    advance = reward - v.data[0]
                    p_loss += - log * advance
                    v_loss += F.smooth_l1_loss(v, var(torch.Tensor([reward])))

                if only_value:
                    loss = v_loss
                else:
                    loss = p_loss + self.alpha * v_loss

                yield [('ploss', p_loss.data[0] / len(saved)),
                       ('vloss', v_loss.data[0] / len(saved))]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                saved = []
                rewards = []

            self.exp_rate *= 0.9995
