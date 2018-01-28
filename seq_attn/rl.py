import torch
import logging
import torch.nn.functional as F
import numpy as np

from .util import var, save_snapshot, progress, normalize


class DDPG:
    def __init__(self, model):
        self.model = model
        self.gamma = 0.95
        self.alpha = 100
        self.warmup = 3

    @progress(print_every=1, logger=logging.getLogger())
    def reinforce(self, data, chars, batch_size=8):
        model = self.model
        actor_optim = torch.optim.Adam(model.policy.parameters(),
                                       weight_decay=0.1)
        critic_optim = torch.optim.Adam(model.value.parameters(),
                                        weight_decay=0.1)

        yield len(data.keys) // batch_size

        N = 0
        n = 0
        p_loss = 0.
        v_loss = 0.
        rs = []
        for key, item in data.epoch(1, backend='torch'):
            N += 1
            L = item.y.size(1) * 2 + 20

            sentence = item.y
            L = sentence.size(1) + 1

            null = torch.zeros(1, 1).type_as(sentence)
            beg = torch.zeros(1, 1).type_as(sentence) + 1

            x = var(torch.cat([beg, sentence], dim=1),
                    volatile=True).permute(1, 0)
            y_true = var(torch.cat([sentence, null], dim=1).permute(1, 0))

            h, h_img, action = model.get_initial_state(var(item.file))

            # value = None
            for i in range(L):
                n += 1

                # env state
                y_, h, alpha, c = model(x[i:i + 1, :], h, h_img, action)
                c.volatile = False

                action = model.policy(c)
                value_ = model.value(torch.cat([c, action],
                                               dim=1)).view(1)

                out = y_.data.squeeze().max(0)[1]
                if out[0] == y_true.data[i, 0]:
                    r = 1.
                else:
                    r = 0.
                rs.append(r)

                p_loss += value_

                # if i > 0:
                #     target_value = var((value_ * self.gamma + r).data)
                #     v_loss += F.smooth_l1_loss(value, target_value)
                # else:
                # r = F.cross_entropy(y_.squeeze(1), y_true[i])
                r = F.softmax(y_, dim=1).squeeze()[y_true[i]]
                v_loss += F.smooth_l1_loss(value_, var(r.data))

                # value = value_

            if N % batch_size == 0:
                critic_optim.zero_grad()
                v_loss.backward(retain_graph=True)
                critic_optim.step()

                if self.warmup == 0:
                    actor_optim.zero_grad()
                    p_loss.backward()
                    actor_optim.step()

                yield [('ploss', p_loss.data[0] / n),
                       ('vloss', v_loss.data[0] / n),
                       ('reward', np.mean(rs))]

                n = 0
                p_loss = 0.
                v_loss = 0.
                rs = []

        if self.warmup > 0:
            self.warmup -= 1
