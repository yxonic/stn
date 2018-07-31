import json
import torch
from collections import namedtuple
import os
import logging
import contextlib
import functools
import time
import sys
from collections import defaultdict, deque

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize(x):
    if x.size(0) <= 1:
        return x
    return (x - x.mean()) / (x.std() + 1e-9)


def save_config(obj, path):
    f = open(path, 'w')
    json.dump(obj.args._asdict(), f)
    f.close()


def load_config(Model, path):
    setup = json.load(open(path, 'r'))
    setup = namedtuple('Setup', setup.keys())(*setup.values())
    return Model(setup)


def var(*args, **kwargs):
    v = torch.autograd.Variable(*args, **kwargs)
    if use_cuda:
        v = v.cuda()
    return v


def save_snapshot(model, ws, id):
    filename = os.path.join(ws, 'snapshots', 'model.%s' % str(id))
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def load_snapshot(model, ws, id):
    filename = os.path.join(ws, 'snapshots', 'model.%s' % str(id))
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def load_last_snapshot(model, ws):
    last = 0
    for file in os.listdir(os.path.join(ws, 'snapshots')):
        if 'model.' in file:
            try:
                epoch = int(file.split('.')[1])
            except ValueError:
                continue
            if epoch > last:
                last = epoch
    if last > 0:
        load_snapshot(model, ws, last)
    return last


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored(text, color, bold=False):
    if bold:
        return bcolors.BOLD + color + text + bcolors.ENDC
    else:
        return color + text + bcolors.ENDC


LOG_COLORS = {
    'WARNING': bcolors.WARNING,
    'INFO': bcolors.OKGREEN,
    'DEBUG': bcolors.OKBLUE,
    'CRITICAL': bcolors.WARNING,
    'ERROR': bcolors.FAIL
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, datefmt, use_color=True):
        logging.Formatter.__init__(self, msg, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in LOG_COLORS:
            record.levelname = colored(record.levelname[0],
                                       LOG_COLORS[record.levelname])
        return logging.Formatter.format(self, record)


@contextlib.contextmanager
def chdir(dir):
    curdir = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(curdir)


def progress(time_interval=None, print_every=None, logger=None):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            then = time.time()
            values = defaultdict(float)
            counts = defaultdict(int)
            n = 0
            dn = 0
            N = 'unk'
            for data in func(*args, **kwargs):
                if isinstance(data, int):
                    # reset
                    values = defaultdict(float)
                    counts = defaultdict(int)
                    N = data
                    continue
                dn += 1
                n += 1
                for k, v in data:
                    if k == 'n':
                        dn += v - 1
                        n += v - 1
                    values[k] += v
                    counts[k] += 1

                if not (print_every is None and time_interval is None):
                    if print_every is not None and n % print_every != 0:
                        if n != N:
                            continue
                    now = time.time()
                    if time_interval is not None and \
                            now - then < time_interval:
                        if n != N:
                            continue

                msg = []
                for k in counts:
                    msg.append('{}: {:.4g}'.format(k, values[k] / counts[k]))

                msg = ', '.join(msg)

                msg = '[{}/{}] ({:.2f}/min) ' \
                    .format(n, N, 60 * dn / (now - then + 1e-9)) + msg

                dn = 0
                then = now

                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg, file=sys.stderr)

        return wrapper

    return decorate
