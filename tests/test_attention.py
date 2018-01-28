from collections import namedtuple

import torch
from torch.autograd import Variable as var
from torchvision.transforms import ToTensor

from yata.fields import Image, Categorical

from seq_attn.model import Attn
from seq_attn.dataprep import load_img


def test_vision():
    setup = {
        "words": "data/words.txt",
        "img_size": 128,
        "emb_size": 128,
        "hidden_size": 128,
        "vision": "vgg"
    }

    setup = namedtuple('Setup', setup.keys())(*setup.values())
    model = Attn(setup)

    img = var(load_img('tests/test_img/001.png', (256, 128)).unsqueeze(0))

    cat = Categorical()
    dict = open('data/words.txt').read().strip().split('\n')
    cat.load_dict(dict)
    x = cat.apply(None, 'y = \frac { 1 } { x }'.split(' '))
    x = var(torch.LongTensor(x).view(-1, 1))

    h, h_img, s = model.get_initial_state(img)

    assert h.size() == (1, 1, 128)
    assert h_img.size() == (1, 16, 32, 128)
    assert s.size() == (1, 3)

    s[0, 0] = 0.15
    s[0, 1] = 0.15
    s[0, 2] = -2
    _, _, alpha, s, advantage = model(x[0], h, h_img, s)

    assert alpha.size() == (1, 16, 32)
    assert s.size() == (1, 3)
    assert advantage.size() == (1, 1)

    _, _, alpha = model(x[0], h, h_img)

    # import matplotlib.pyplot as plt
    # plt.matshow(alpha[0].data.numpy())
    # plt.show()


if __name__ == '__main__':
    test_vision()
