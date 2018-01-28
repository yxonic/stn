import torch
from torch.autograd import Variable as var

from seq_attn.model import SimpleVGG, SimpleResNet


def test_vision():
    vgg = SimpleVGG(128)
    res = SimpleResNet()

    x = var(torch.rand(8, 3, 256, 256))

    assert vgg(x).size() == (8, 128, 32, 32)
    assert res(x).size() == (8, 128, 32, 32)
