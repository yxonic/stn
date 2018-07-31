"""Prepare data for training."""

# TODO: discard yata...
from yata.loaders import DirectoryLoader, TableLoader, DataLoader
from yata.fields import Image, Chars, Words, Categorical

from torchvision.transforms import ToTensor


def get_kdd_dataset(name, large_image=False):
    r"""Load datasets for KDD paper.

    Args:
        name(str): choose from formula, melody and multiline
        large_image(bool): larger image size (Default: ``False``)
    """
    cat = Categorical()
    dict = open('data/%s_words.txt' % name).read().strip().split('\n')
    cat.load_dict(dict)
    labels = TableLoader('data/%s_label.txt' % name, key='uuid',
                         fields={'label->y': cat(Words(' '))})

    if large_image:
        if name == 'melody':
            imgs = DirectoryLoader('data/' + name, Image((128, 64), True))
        elif name == 'formula':
            imgs = DirectoryLoader('data/' + name, Image((256, 128), True))
        else:
            imgs = DirectoryLoader('data/' + name, Image((256, 256), True))
    else:
        if name == 'melody':
            imgs = DirectoryLoader('data/' + name, Image((128, 32), True))
        elif name == 'formula':
            imgs = DirectoryLoader('data/' + name, Image((128, 64), True))
        else:
            imgs = DirectoryLoader('data/' + name, Image((64, 64), True))

    data = DataLoader(imgs, labels)

    return data, cat


def get_recog(name):
    if name.startswith('iiit'):
        return get_iiit5k(name)
    cat = Categorical()
    dict = open('data/chars.txt').read().strip().split('\n')
    cat.load_dict(dict)
    labels = TableLoader('data/%s_label.txt' % name, key='uuid',
                         fields={'label->y': cat(Chars())})

    imgs = DirectoryLoader('data/' + name, Image((128, 64)))
    data = DataLoader(imgs, labels)
    return data, cat


def get_iiit5k(name):
    cat = Categorical()
    dict = open('data/chars.txt').read().strip().split('\n')
    cat.load_dict(dict)
    labels = TableLoader('data/%s_label.txt' % name, key='uuid',
                         fields={'label->y': cat(Chars()),
                                 'pos': Words(';')})

    imgs = DirectoryLoader('data/' + name, Image((128, 64)))
    data = DataLoader(imgs, labels)
    return data, cat


def get_formula(label=False, use_token=True):
    cat = Categorical()
    if use_token:
        dict = open('data/formula_words.txt').read().strip().split('\n')
    else:
        dict = open('data/chars.txt').read().strip().split('\n')
    cat.load_dict(dict)

    if use_token:
        labels = TableLoader('data/formula_label_simple.txt', key='uuid',
                             fields={'latex->y': cat(Words(' '))})
    else:
        labels = TableLoader('data/label.txt', key='uuid',
                             fields={'latex->y': cat(Chars())})

    # if label:
    #     return labels, cat

    imgs = DirectoryLoader('data/formula', Image((256, 128), True))
    data = DataLoader(imgs, labels)
    return data, cat


def get_melody():
    cat = Categorical()
    dict = open('data/melody_chars.txt').read().strip().split('\n')
    cat.load_dict(dict)

    labels = TableLoader('data/melody_label_simple.txt', key='uuid',
                         fields={'label->y': cat(Words(' '))})

    # if label:
    #     return labels, cat

    imgs = DirectoryLoader('data/melody', Image((256, 128), True))
    data = DataLoader(imgs, labels)
    return data, cat


def get_cat(dict_file):
    cat = Categorical()
    dict = open(dict_file).read().strip().split('\n')
    cat.load_dict(dict)
    return cat


def load_img(filename, size, gray_scale):
    im = Image(size, gray_scale).apply(None, filename)
    return ToTensor()(im)


if __name__ == '__main__':
    data, cat = get_formula(use_token=False)
    avg = []
    for key in data.keys:
        avg.append(len(data.get(key).y))
    import numpy as np
    print(np.mean(avg))
