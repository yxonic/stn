import time
import logging
from collections import deque

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import *

from . import dataprep
from .rl import *
from .util import *
from .agent import *


def train(model, args):
    r"""Train the model."""
    model.train()
    if args.snapshot is None:
        start_epoch = 0
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
        logging.info('Loaded ' + str(epoch))
        start_epoch = int(epoch.split('.')[-1])

    if args.dataset == 'multiline':
        agent_hs = 128
    else:
        agent_hs = 64

    if args.spotlight_model == 'markov':
        agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
    elif args.spotlight_model == 'rnn':
        agent = RNNPolicy(model.hidden_size + model.img_size + 3, agent_hs)
    else:
        agent = None

    if args.focus is not None:
        load_snapshot(agent, args.workspace, args.focus)
        logging.info('Loaded ' + str(args.focus))

    logging.info('Loading data...')
    # if args.dataset == 'melody':
    #     data, cat = dataprep.get_melody()
    # elif args.dataset == 'formula':
    #     data, cat = dataprep.get_formula()
    # else:
    #     data, cat = dataprep.get_recog(args.dataset)

    data, cat = dataprep.get_kdd_dataset(args.dataset, args.spotlight_model)

    if args.split_frac > 0:
        data, _ = data.split(args.split_frac)

    if use_cuda:
        model.cuda()
        if agent:
            agent.cuda()

    if agent:
        optim = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.norm)
        optim_output = torch.optim.Adam(model.output.parameters(),
                                        weight_decay=args.norm)
        agent_optim = torch.optim.Adam(agent.parameters(),
                                       weight_decay=args.norm)
    else:
        optim = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.norm)

    for epoch in range(start_epoch, args.epochs):
        N = 0
        n = 0
        total_loss = 0.
        total_agent_loss = 0.
        total = len(data.keys)
        for keys, item in data.shuffle().epoch(args.batch_size,
                                               backend='torch'):
            N += len(keys)

            if type(item.y) != tuple:
                sentences = item.y
                lens = [item.y.size(1)] * item.y.size(0)
            else:
                sentences, lens = item.y

            hs, h_imgs, ss = model.get_initial_state(var(item.file))

            loss = 0.
            agent_loss = 0.
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
                        if len(item) == 5:
                            if i != L:
                                xx, yy = item.pos[k][i][1:-1].split(',')
                                xx = float(xx)
                                yy = float(yy)
                                sigma = -3.
                                s_true = torch.Tensor([[xx, yy, sigma]])
                            else:
                                s_true = s.data
                            s_pred, sh = agent(s, var(c.data), sh)
                            y_pred, h, alpha, c = \
                                model.put_h(h, h_img, var(s_true))
                            agent_loss += F.smooth_l1_loss(s_pred, var(s_true))
                            s = var(s_true)
                        else:
                            s, sh = agent(s, c, sh)
                            y_pred, h, alpha, c = model.put_h(h, h_img, s)
                    else:
                        y_pred, h, alpha, _ = model(x[i:i + 1, :], h, h_img)
                    loss += F.cross_entropy(y_pred, y_true[i])

            total_loss += loss.data[0]
            if type(agent_loss) != float:
                total_agent_loss += agent_loss.data[0]

            optim.zero_grad()
            if agent:
                agent_optim.zero_grad()

            loss.backward()
            if type(agent_loss) != float:
                agent_loss.backward()

            optim.step()
            if agent:
                agent_optim.step()

            loss = 0.
            agent_loss = 0.

            now = time.time()
            duration = (now - then) / 60
            logging.info('[%d:%d/%d] (%.2f samples/min) '
                         'loss %.6f, agent_loss %.6f' %
                         (epoch, N, total, args.batch_size / duration,
                          total_loss / n,
                          total_agent_loss / n))

        if False:
            for keys, item in data.sample(0.05).epoch(args.batch_size,
                                                      backend='torch'):
                N += len(keys)

                if type(item.y) != tuple:
                    sentences = item.y
                    lens = [item.y.size(1)] * item.y.size(0)
                else:
                    sentences, lens = item.y

                hs, h_imgs, ss = model.get_initial_state(var(item.file))

                loss = 0.
                agent_loss = 0.
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
                            if len(item) == 5:
                                if i != L:
                                    xx, yy = item.pos[k][i][1:-1].split(',')
                                    xx = float(xx)
                                    yy = float(yy)
                                    sigma = -3.
                                    s_true = torch.Tensor([[xx, yy, sigma]])
                                else:
                                    s_true = s.data
                                s_pred, sh = agent(s, var(c.data), sh)
                                y_pred, h, alpha, c = \
                                    model.put_h(h, h_img, var(s_true))
                                agent_loss += F.smooth_l1_loss(s_pred, var(s_true))
                                s = var(s_true)
                            else:
                                s, sh = agent(s, c, sh)
                                y_pred, h, alpha, c = model.put_h(h, h_img, s)
                        else:
                            y_pred, h, alpha, _ = model(x[i:i + 1, :], h, h_img)
                        loss += F.cross_entropy(y_pred, y_true[i])

                total_loss += loss.data[0]
                if type(agent_loss) != float:
                    total_agent_loss += agent_loss.data[0]

                optim_output.zero_grad()
                agent_optim.zero_grad()

                loss.backward()
                if type(agent_loss) != float:
                    agent_loss.backward()

                optim_output.step()
                agent_optim.step()

                loss = 0.
                agent_loss = 0.

                now = time.time()
                duration = (now - then) / 60
                logging.info('[%d:%d/%d] (%.2f samples/min) '
                             'loss %.6f, agent_loss %.6f' %
                             (epoch, N, total, args.batch_size / duration,
                              total_loss / n,
                              total_agent_loss / n))

        save_snapshot(model, args.workspace, 'main.' + str(epoch + 1))
        if agent:
            save_snapshot(agent, args.workspace, 'agent.' + str(epoch + 1))


def train_batched(model, args):
    model.train()
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    # if args.dataset == 'melody':
    #     data, _ = dataprep.get_melody()
    # else:
    #     data, _ = dataprep.get_formula()

    # data, _ = data.split(args.split_frac)
    data, cat = dataprep.get_kdd_dataset(args.dataset)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.norm)

    start_epoch = load_last_snapshot(model, args.workspace)

    if use_cuda:
        model.cuda()

    for epoch in range(start_epoch, args.epochs):
        try:
            logging.info(('epoch {}:'.format(epoch)))
            N = len(data.keys)
            n = 0
            total_loss = 0.
            then = time.time()
            time_n = 0
            for keys, item in data.shuffle().epoch(args.batch_size,
                                                   backend='torch'):
                if len(item.y) != 2:
                    sentences = item.y
                    lens = [item.y.size(1)] * item.y.size(0)
                else:
                    sentences, lens = item.y

                plens = [l + 1 for l in lens]

                bz = len(keys)
                nulls = torch.zeros(bz, 1).type_as(sentences)
                begs = torch.zeros(bz, 1).type_as(sentences) + 1

                x = var(torch.cat([begs, sentences],
                                  dim=1)).permute(1, 0)
                y_true = var(torch.cat([sentences, nulls],
                                       dim=1)).permute(1, 0)
                y_true_ = pack_padded_sequence(y_true, plens)

                y_pred_ = model.pred_on_batch(var(item.file), x, plens)

                loss = F.cross_entropy(y_pred_, y_true_.data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n += len(keys)
                time_n += len(keys)
                total_loss += loss.data[0] * len(keys)

                now = time.time()
                if now - then < 5:
                    continue

                duration = (now - then) / 60
                logging.info('[%d:%d/%d] (%.2f samples/min) '
                             'loss %.6f' %
                             (epoch, n, N, time_n / duration,
                              total_loss / n))
                then = now
                time_n = 0

            save_snapshot(model, args.workspace, epoch + 1)

        except KeyboardInterrupt as e:
            save_snapshot(model, args.workspace, 'int')
            raise e


def test(model, args):
    model.eval()
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    logging.info('Loading data...')

    # if args.dataset == 'melody':
    #     data, cat = dataprep.get_melody()
    # elif args.dataset == 'formula':
    #     data, cat = dataprep.get_formula()
    # else:
    #     data, cat = dataprep.get_recog(args.dataset)

    data, cat = dataprep.get_kdd_dataset(args.dataset, args.focus)

    if args.split_frac > 0:
        _, data = data.split(args.split_frac)

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)

    if args.dataset == 'multiline':
        agent_hs = 128
    else:
        agent_hs = 64

    if args.focus:
        if args.spotlight_model == 'markov':
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        else:
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, agent_hs)
        load_snapshot(agent, args.workspace, args.focus)

    logging.info('loaded model at epoch %s', str(epoch))

    if use_cuda:
        model.cuda()

    N = len(data.keys)
    n = 0
    total_acc = 0.

    for keys, item in data.shuffle().epoch(1, backend='torch'):
        n += 1

        sentence = item.y
        L = sentence.size(1)

        null = torch.zeros(1, 1).type_as(sentence)
        beg = torch.zeros(1, 1).type_as(sentence) + 1

        x = var(torch.cat([beg, sentence], dim=1), volatile=True).permute(1, 0)
        y_true = torch.cat([sentence, null], dim=1).permute(1, 0)
        y_pred = var(torch.zeros(L * 2 + 21, 1).type(torch.LongTensor))
        img = var(item.file, volatile=True)
        h, h_img, s = model.get_initial_state(img)

        same = 0
        y_ = x[0:1, :]

        if args.focus:
            sh = agent.default_h()
            c = agent.default_c()

        true_seq = []
        pred_seq = []
        for i in range(L * 2 + 20):
            if args.focus:
                h = model.get_h(y_, h)
                c = torch.cat([h.view(1, -1), c[:, model.hidden_size:]], dim=1)
                s, sh = agent(s, c, sh)
                y_, h, alpha, c = model.put_h(h, h_img, s)
            else:
                y_, h, _, _ = model(y_, h, h_img)
            y_ = y_.max(1)[1]
            y_pred[i, 0] = y_
            if i <= L:
                true_seq.append(y_true[i, 0])
                same += y_.data[0] == y_true[i, 0]
            pred_seq.append(y_.data[0])
            y_ = y_.unsqueeze(0)
        total_acc += lcs(true_seq, pred_seq) / len(true_seq) \
            if args.lcs else same / (L + 1)
        if n % 50 == 0 or n == len(data.keys):
            logging.info('[%d/%d] acc %.6f' %
                         (n, N, total_acc / n))


def lcs(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j],
                                            lengths[i][j + 1])
    # read the substring out from the matrix
    result = 0
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result += 1
            x -= 1
            y -= 1
    return result


def examine(model, args):
    model.eval()
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)

    if args.dataset == 'melody':
        data, chars = dataprep.get_melody()
    else:
        data, chars = dataprep.get_formula()
    _, data = data.split(0.9)

    if args.focus:
        if args.spotlight_model == 'markov':
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        else:
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, 32)
        load_snapshot(agent, args.workspace, args.focus)

    if use_cuda:
        model.cuda()

    n = 0

    for keys, item in data.shuffle().epoch(1, backend='torch'):
        if n == 50:
            break
        n += 1

        sentence = item.y
        L = sentence.size(1)

        null = torch.zeros(1, 1).type_as(sentence)
        beg = torch.zeros(1, 1).type_as(sentence) + 1

        x = var(torch.cat([beg, sentence], dim=1), volatile=True).permute(1, 0)
        y_true = torch.cat([sentence, null], dim=1).permute(1, 0)
        y_pred = var(torch.zeros(L + 21, 1).type(torch.LongTensor))

        h, h_img, s = model.get_initial_state(var(item.file, volatile=True))
        if args.focus:
            sh = agent.default_h()
            c = agent.default_c()

        if args.focus:
            s, sh = agent(s, c, sh)
            y_, h, alpha, c = model(x[0:1, :], h, h_img, s)
        else:
            y_, h, alpha = model(x[0:1, :], h, h_img)
        y_pred[0, 0] = y_.squeeze().max(0)[1]

        same = y_pred.data[0, 0] == y_true[0, 0]
        for i in range(L + 20):
            if args.focus:
                s, sh = agent(s, c, sh)
                y_, h, alpha, c = model(y_pred[i:i + 1, :], h, h_img, s)
            else:
                y_, h, alpha = model(y_pred[i:i + 1, :], h, h_img)
            out = y_.squeeze().max(0)[1]
            y_pred[i + 1, 0] = out
            if same and out.data[0] != y_true[i + 1, 0]:
                same = False
            if out.data[0] == 0:
                break

        if same:
            color = bcolors.OKGREEN
        else:
            color = bcolors.FAIL

        print(colored('> ', bcolors.OKGREEN, True) +
              ''.join(chars.get_original(y_true[:-1].squeeze())))
        print(colored('< ', color, True) +
              ''.join(chars.get_original(y_pred.squeeze()[:i + 1].data)))
        print()


def visualize(model, args):
    model.eval()
    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
        logging.info('Loaded epoch: ' + epoch)

    agent_hs = args.agent_hs
    if args.focus:
        if args.spotlight_model == 'markov':
            agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
        else:
            agent = RNNPolicy(model.hidden_size + model.img_size + 3, agent_hs)
        load_snapshot(agent, args.workspace, args.focus)

    cat = dataprep.get_cat(model.args.words)
    img = dataprep.load_img(args.input, (args.W, args.H),
                            not args.colored).unsqueeze(0)
    img = var(img, volatile=True)
    h, h_img, s = model.get_initial_state(img)
    if args.focus:
        sh = agent.default_h()
        c = agent.default_c()
    x = var(torch.zeros(1, 1).long() + 1)
    import matplotlib.pyplot as plt

    # print(next(model.focus_trans.parameters())[0].data.numpy())
    for _ in range(40):
        # y, h, alpha = model(x, h, h_img)
        if args.focus:
            h = model.get_h(x, h)
            c = torch.cat([h.view(1, -1), c[:, model.hidden_size:]], dim=1)
            s, sh = agent(s, c, sh)
            y, h, alpha, c = model.put_h(h, h_img, s)
        else:
            y, h, alpha, _ = model(x, h, h_img)

        if args.focus:
            v = torch.zeros(3)
            v[:] = s.squeeze().data
            v[2:] = torch.sigmoid(v[2:])
            # v.clamp_(0, 1)
            print(*v.numpy())
        y = y.data.max(1)[1]
        print(cat.get_original(y)[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #a = -F.sigmoid(alpha[0] / 10)
        a = -alpha[0]
        cax = ax.matshow(a.data.numpy(), cmap='RdYlBu',
                         interpolation='gaussian')
        fig.colorbar(cax)
        fig.show()
        plt.show()
        if y[0] == 0:
            break
        x = var(y).unsqueeze(0)


def reinforce(model, args):
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))

    # load pretrained model
    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('Loaded epoch: ' + str(epoch))
    logging.info('Loading data...')

    if args.dataset == 'melody':
        data, cat = dataprep.get_melody()
    else:
        data, cat = dataprep.get_formula()
    data, _ = data.split(0.9)

    learner = ActorCritic(model)

    if use_cuda:
        model.cuda()

    for i in range(args.times):
        logging.info('start %dth time', i)
        learner.reinforce(data.shuffle(), cat)
        save_snapshot(model, args.workspace, 'rf.' + str(i))
        # TODO: validate


# deprecated
def pretrain_attention(model, args):
    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('Loaded epoch: ' + str(epoch))

    if args.spotlight_model == 'markov':
        agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
    else:
        agent = RNNPolicy(model.hidden_size + model.img_size + 3, 32)

    logging.info('Loading data...')
    if args.dataset == 'melody':
        data, cat = dataprep.get_melody()
    else:
        data, cat = dataprep.get_formula()
    data, _ = data.split(args.split_frac)

    if use_cuda:
        model.cuda()
        agent.cuda()

    model.suspend_focus = True

    optim = torch.optim.Adam(model.parameters(), weight_decay=args.norm)
    agent_optim = torch.optim.Adam(agent.parameters(), weight_decay=args.norm)

    for epoch in range(args.epochs):
        N = 0
        loss = 0.
        agent_loss = 0.

        total_loss = 0.
        total_agent_loss = 0.
        n = 0
        then = time.time()

        for keys, item in data.shuffle().epoch(1, backend='torch'):
            N += 1

            sentence = item.y
            L = sentence.size(1)

            null = torch.zeros(1, 1).type_as(sentence)
            beg = torch.zeros(1, 1).type_as(sentence) + 1
            x = var(torch.cat([beg, sentence], dim=1)).permute(1, 0)
            y_true = var(torch.cat([sentence, null], dim=1).permute(1, 0))

            h, h_img, s = model.get_initial_state(var(item.file))
            sh = agent.default_h()
            c = agent.default_c()

            for i in range(L + 1):
                n += 1
                s_pred, sh = agent(s, var(c.data), sh)
                y_pred, h, alpha, c = model(x[i:i + 1, :], h, h_img)
                loss += F.cross_entropy(y_pred, y_true[i])

                s = var(c[:, -3:].data)
                agent_loss += F.smooth_l1_loss(s_pred, s)

            if N % args.batch_size == 0:
                total_loss += loss.data[0]
                total_agent_loss += agent_loss.data[0]

                optim.zero_grad()
                loss.backward()
                optim.step()

                agent_optim.zero_grad()
                agent_loss.backward()
                agent_optim.step()

                loss = 0.
                agent_loss = 0.

                now = time.time()
                duration = (now - then) / 60
                logging.info('[%d:%d] (%.2f samples/min) '
                             'loss %.6f, agent_loss %.6f' %
                             (epoch, N, args.batch_size / duration,
                              total_loss / n,
                              total_agent_loss / n))
                then = now

        save_snapshot(model, args.workspace, 'focus.' + str(epoch + 1))
        save_snapshot(agent, args.workspace, 'agent.pre.' + str(epoch + 1))


def train_backup(model, args):
    if args.snapshot is None:
        start_epoch = 0
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
        logging.info('Loaded ' + str(epoch))
        start_epoch = int(epoch.split('.')[-1])

    if args.spotlight_model == 'markov':
        agent = MarkovPolicy(model.hidden_size + model.img_size + 3)
    elif args.spotlight_model == 'rnn':
        agent = RNNPolicy(model.hidden_size + model.img_size + 3, 32)
    else:
        agent = None

    if args.focus is not None:
        load_snapshot(agent, args.workspace, args.focus)
        logging.info('Loaded ' + str(args.focus))

    logging.info('Loading data...')
    if args.dataset == 'melody':
        data, cat = dataprep.get_melody()
    elif args.dataset == 'formula':
        data, cat = dataprep.get_formula()
    else:
        data, cat = dataprep.get_recog(args.dataset)
    if args.split_frac > 0:
        data, _ = data.split(args.split_frac)

    if use_cuda:
        model.cuda()
        if agent:
            agent.cuda()

    if agent:
        optim = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.norm)
        agent_optim = torch.optim.Adam(agent.parameters(),
                                       weight_decay=args.norm)
    else:
        optim = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.norm)

    for epoch in range(start_epoch, args.epochs):
        N = 0
        loss = 0.
        agent_loss = 0.
        total_loss = 0.
        total_agent_loss = 0.
        n = 0
        then = time.time()

        for keys, item in data.shuffle().epoch(1, backend='torch'):
            N += 1

            sentence = item.y
            L = sentence.size(1)

            null = torch.zeros(1, 1).type_as(sentence)
            beg = torch.zeros(1, 1).type_as(sentence) + 1
            x = var(torch.cat([beg, sentence], dim=1)).permute(1, 0)
            y_true = var(torch.cat([sentence, null], dim=1).permute(1, 0))

            h, h_img, s = model.get_initial_state(var(item.file))

            if agent:
                sh = agent.default_h()
                c = agent.default_c()

            for i in range(L + 1):
                n += 1
                if agent:
                    if len(item) == 5:
                        if i != L:
                            xx, yy = item.pos[0][i][1:-1].split(',')
                            xx = float(xx)
                            yy = float(yy)
                            sigma = -3.
                            s_true = torch.Tensor([[xx, yy, sigma]])
                        else:
                            s_true = s.data
                        s_pred, sh = agent(s, var(c.data), sh)
                        y_pred, h, alpha, c = model(x[i:i + 1, :],
                                                    h, h_img, var(s_true))
                        agent_loss += F.smooth_l1_loss(s_pred, var(s_true))
                        s = var(s_true)
                    else:
                        s_pred, sh = agent(s, var(c.data), sh)
                        y_pred, h, alpha, c = model(x[i:i + 1, :],
                                                    h, h_img, s_pred)
                        s = s_pred
                else:
                    y_pred, h, alpha = model(x[i:i + 1, :], h, h_img)
                loss += F.cross_entropy(y_pred, y_true[i])

            if N % args.batch_size == 0:
                total_loss += loss.data[0]
                if type(agent_loss) != float:
                    total_agent_loss += agent_loss.data[0]

                optim.zero_grad()
                if agent:
                    agent_optim.zero_grad()
                loss.backward()

                if type(agent_loss) != float:
                    agent_loss.backward()

                optim.step()
                if agent:
                    agent_optim.step()

                loss = 0.
                agent_loss = 0.

                now = time.time()
                duration = (now - then) / 60
                logging.info('[%d:%d] (%.2f samples/min) '
                             'loss %.6f, agent_loss %.6f' %
                             (epoch, N, args.batch_size / duration,
                              total_loss / n,
                              total_agent_loss / n))
                then = now

        save_snapshot(model, args.workspace, 'main.' + str(epoch + 1))
        if agent:
            save_snapshot(agent, args.workspace, 'agent.' + str(epoch + 1))
