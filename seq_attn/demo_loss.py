from collections import namedtuple

import sys
import time
import logging
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from yata.fields import Image, Categorical

from .model import Attn, Spotlight
from .dataprep import get_kdd_dataset
from .util import use_cuda, var
from .agent import MarkovPolicy, RNNPolicy


def attn_model(dataset, attn, pos_emb):
    setup = {
        "words": "data/%s_words.txt" % dataset,
        "img_size": 128,
        "emb_size": 32,
        "hidden_size": 128,
        "vision": "resnet",
        "pos_emb": pos_emb,
        "attention": attn
    }

    setup = namedtuple('Setup', setup.keys())(*setup.values())
    model = Attn(setup)
    return model


def spotlight_model(dataset):
    setup = {
        "words": "data/%s_words.txt" % dataset,
        "img_size": 128,
        "emb_size": 32,
        "hidden_size": 128,
        "vision": "resnet",
        "attention": "none"
    }

    setup = namedtuple('Setup', setup.keys())(*setup.values())
    model = Spotlight(setup)
    return model


def run(model, data, agent=None, optim=None, agent_optim=None, batch_size=16):
    N = 0
    n = 0
    total_loss = 0.
    total = len(data.keys)
    for keys, item in data.shuffle().epoch(batch_size,
                                           backend='torch'):
        N += len(keys)

        if type(item.y) != tuple:
            sentences = item.y
            lens = [item.y.size(1)] * item.y.size(0)
        else:
            sentences, lens = item.y

        if optim is not None:
            img = var(item.file)
        else:
            img = var(item.file, volatile=True)
        hs, h_imgs, ss = model.get_initial_state(img)

        loss = 0.
        then = time.time()

        for k in range(len(keys)):
            sentence = sentences[k:k + 1, :lens[k]]
            L = sentence.size(1)

            null = torch.zeros(1, 1).type_as(sentence)
            beg = torch.zeros(1, 1).type_as(sentence) + 1
            x = var(torch.cat([beg, sentence], dim=1)).permute(1, 0)
            y_true = var(torch.cat([sentence, null], dim=1).permute(1, 0))

            h = hs[:, k:k + 1, :]
            h_img = h_imgs[k:k + 1, :, :, :]
            s = ss[k:k + 1, :] if ss is not None else None

            if agent:
                sh = agent.default_h()
                c = agent.default_c()

            for i in range(L + 1):
                n += 1
                if agent:
                    h = model.get_h(x[i:i + 1, :], h)
                    c = torch.cat([h.view(1, -1),
                                   c[:, model.hidden_size:]], dim=1)
                    s, sh = agent(s, c, sh)
                    y_pred, h, alpha, c = model.put_h(h, h_img, s)
                else:
                    y_pred, h, alpha, _ = model(x[i:i + 1, :], h, h_img)
                loss += F.cross_entropy(y_pred, y_true[i])

        total_loss += loss.data[0]

        if optim is not None:
            optim.zero_grad()
            if agent_optim is not None:
                agent_optim.zero_grad()

            loss.backward()

            optim.step()
            if agent_optim is not None:
                agent_optim.step()

        loss = 0.

        now = time.time()
        duration = (now - then) / 60
        logging.info('[%d/%d] (%.2f samples/min) loss %.6f' %
                     (N, total, batch_size / duration,
                      total_loss / n))

    return total_loss / n, model


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    dataset, model_type = sys.argv[1:]
    if '-' in model_type:
        attn = model_type.split('-')[1]

    data, _ = get_kdd_dataset(dataset)
    if dataset == 'formula':
        trainset, evalset = data.sample(8000).split(0.8)
    elif dataset == 'multiline':
        trainset, evalset = data.shuffle().split(0.9)
    else:
        trainset, evalset = data.sample(2000).split(0.9)

    if model_type.startswith('attn'):
        model = attn_model(dataset, attn, model_type.endswith('pos'))
        agent = None
        optim = torch.optim.Adam(model.parameters())
        agent_optim = None
        if use_cuda:
            model.cuda()
    else:
        model = spotlight_model(dataset)
        if model_type == 'stnr':
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, 64)
        else:
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        optim = torch.optim.Adam(model.parameters())
        out_optim = torch.optim.Adam(model.output.parameters())
        agent_optim = torch.optim.Adam(agent.parameters(), weight_decay=0.1)
        if use_cuda:
            model.cuda()
            agent.cuda()

    for i in range(40):
        eval_loss, model = run(model, evalset, agent=agent)
        logging.info('-' * 50)
        train_loss, model = run(model, trainset, agent=agent,
                                optim=optim, agent_optim=agent_optim)
        if model_type.startswith('stn'):
            _, model = run(model, trainset.sample(0.2), agent=agent,
                           optim=out_optim, agent_optim=agent_optim)
        logging.info('-' * 50)
        print(train_loss, eval_loss)
