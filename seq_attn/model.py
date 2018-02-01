import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import vgg11_bn, resnet18

import logging
from itertools import chain

from .util import var


class Attn(nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--words', default='data/words.txt')
        parser.add_argument('--img_size', '-is', default=128, type=int,
                            help='size of image features')
        parser.add_argument('--emb_size', '-es', default=32, type=int,
                            help='word emb size')
        parser.add_argument('--hidden_size', '-hs', default=128, type=int,
                            help='size of RNN hidden vector')
        parser.add_argument('--vision', '-v', choices=['vgg', 'resnet'],
                            default='vgg', help='vision model')
        parser.add_argument('--attention', '-a', choices=['none', 'dot', 'fc'],
                            default='dot', help='attention strategy')
        parser.add_argument('--pos_emb', '-p', action='store_true',
                            help='add position embedding layers')

    def __init__(self, args):
        super().__init__()
        self.args = args

        # args
        dict = open(args.words).read().strip().split('\n')
        self.wcnt = len(dict)
        self.img_size = args.img_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.attention = args.attention
        self.pos_emb = args.pos_emb

        # initialize layers
        if args.vision == 'vgg':
            self.vision_model = SimpleVGG(self.img_size)
        else:
            if args.img_size != 128:
                logging.warn('ResNet feature size is fixed at 128')
            self.vision_model = SimpleResNet()

        if self.pos_emb:
            self.img_size += 2

        self.emb = nn.Embedding(self.wcnt, self.emb_size)
        self.rnn = nn.GRU(self.emb_size + self.img_size, self.hidden_size)

        # self.attn = nn.Linear(self.hidden_size + self.img_size, 1)
        if self.attention == 'fc':
            self.attn = FF(self.hidden_size + self.img_size)

        self.output = nn.Linear(self.hidden_size, self.wcnt)

    def get_img_emb(self, img):
        """
        Inputs: img (batch, channel, H, W)
        Outputs: h_img (batch, H // 8, W // 8, img_size)
        """
        emb = self.vision_model(img).permute(0, 2, 3, 1)
        if self.pos_emb:
            sz = list(emb.size())
            H = sz[1]
            W = sz[2]
            sz[3] = 1
            i_layer = var(torch.arange(H) / H).view(1, H, 1, 1).expand(*sz)
            j_layer = var(torch.arange(W) / W).view(1, 1, W, 1).expand(*sz)
            emb = torch.cat([emb, i_layer, j_layer], dim=3)
        return emb

    def get_initial_state(self, img):
        """
        Inputs: img (batch, channel, height, width)
        Outputs: (h_0 (1, batch, hidden_size),
                  h_img (batch, h, w, img_size),
                  s (batch, 3))
        """
        return (var(torch.zeros(1, img.size(0), self.hidden_size)),
                self.get_img_emb(img).contiguous(),
                None)

    def forward(self, x, h_, h_img, focus_=None):
        """
        input one char, output next
        """
        z = self.emb(x.view(1, 1))
        h_attn, alpha = self._get_context(h_.squeeze(0), h_img)
        _, h = self.rnn(torch.cat([z, h_attn.view(-1, 1,
                                                  self.img_size)], dim=2), h_)
        y = self.output(h.squeeze(0))
        return y, h, alpha, h.squeeze(0)

    def _get_context(self, hs, h_imgs):
        """
        Inputs: hs (_, hid_size),
                h_imgs (_, H, W, img_size)
        Outputs: h_attn (_, img_size), alpha (_, n)
        """

        isz = h_imgs.size(-1)
        hsz = hs.size(-1)
        n = h_imgs.size(1) * h_imgs.size(2)

        if self.attention == 'none':
            a = var(torch.zeros(h_imgs.size(0), n))
        elif self.attention == 'fc':
            target_sz = list(hs.size())
            target_sz.insert(1, n)
            hs_ = hs.unsqueeze(1).expand(target_sz).contiguous()

            # image information
            a = self.attn(torch.cat([hs_.view(-1, hsz),
                                     h_imgs.view(-1, isz)],
                                    dim=1)).view(-1, n)
        elif self.attention == 'dot':
            a = torch.bmm(h_imgs.view(-1, n, isz), hs.unsqueeze(-1))
            a = a.squeeze(2)

        if self.attention == 'none':
            size = (h_imgs.size(0), h_imgs.size(1) * h_imgs.size(2),
                    h_imgs.size(3))
            alpha = torch.exp(a).view(-1, n, 1)
            h_attn = alpha.expand(*size) * h_imgs.view(*size)
            h_attn = h_attn.max(1)[0]
        else:
            alpha = F.softmax(a, dim=1).view(-1, 1, n)
            h_attn = torch.bmm(alpha, h_imgs.view(-1, n, isz)).view(-1, isz)

        # output
        return h_attn, alpha.squeeze(1) \
            .view(-1, h_imgs.size(1), h_imgs.size(2))


class Spotlight(nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--words', default='data/words.txt')
        parser.add_argument('--img_size', '-is', default=128, type=int,
                            help='size of image features')
        parser.add_argument('--emb_size', '-es', default=32, type=int,
                            help='word emb size')
        parser.add_argument('--hidden_size', '-hs', default=128, type=int,
                            help='size of RNN hidden vector')
        parser.add_argument('--vision', '-v', choices=['vgg', 'resnet'],
                            default='vgg', help='vision model')
        parser.add_argument('--attention', '-a', choices=['none', 'dot', 'fc'],
                            default='dot', help='attention strategy')

    def __init__(self, args):
        super().__init__()
        self.args = args

        # args
        dict = open(args.words).read().strip().split('\n')
        self.wcnt = len(dict)
        self.img_size = args.img_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.attention = args.attention

        # initialize layers
        if args.vision == 'vgg':
            self.vision_model = SimpleVGG(self.img_size)
        else:
            if args.img_size != 128:
                logging.warn('ResNet feature size is fixed at 128')
            self.vision_model = SimpleResNet()

        self.img_size += 2

        self.emb = nn.Embedding(self.wcnt, self.emb_size)
        self.rnn = nn.GRU(self.emb_size, self.hidden_size)

        # self.attn = nn.Linear(self.hidden_size + self.img_size, 1)
        if self.attention == 'fc':
            self.attn = FF(self.hidden_size + self.img_size)

        # self.policy = nn.Linear(self.hidden_size + self.img_size + 3, 3)
        # self.value = nn.Linear(self.hidden_size + self.img_size + 6, 1)
        self.output = nn.Linear(self.hidden_size + self.img_size + 3,
                                self.wcnt)

    def get_img_emb(self, img):
        """
        Inputs: img (batch, channel, H, W)
        Outputs: h_img (batch, H // 8, W // 8, img_size)
        """
        emb = self.vision_model(img).permute(0, 2, 3, 1)
        sz = list(emb.size())
        H = sz[1]
        W = sz[2]
        sz[3] = 1
        i_layer = var(torch.arange(H) / H).view(1, H, 1, 1).expand(*sz)
        j_layer = var(torch.arange(W) / W).view(1, 1, W, 1).expand(*sz)
        emb = torch.cat([emb, i_layer, j_layer], dim=3)
        return emb

    def get_initial_state(self, img):
        """
        Inputs: img (batch, channel, height, width)
        Outputs: (h_0 (1, batch, hidden_size),
                  h_img (batch, h, w, img_size),
                  s (batch, 3))
        """
        return (var(torch.zeros(1, img.size(0), self.hidden_size)),
                self.get_img_emb(img).contiguous(),
                var(torch.zeros(img.size(0), 3)))

    def forward(self, x, h_, h_img, focus=None):
        """
        input one char, output next
        """
        z = self.emb(x.view(1, 1))
        _, h = self.rnn(z, h_)
        h_attn, alpha = self._get_context(h.squeeze(0), h_img, focus)
        if focus is None:
            focus = get_handle(alpha)
        c = torch.cat([h.squeeze(0), h_attn, focus], dim=1)
        y = self.output(c)
        return y, h, alpha, c

    def put_h(self, h, h_img, focus=None):
        h_attn, alpha = self._get_context(h.squeeze(0), h_img, focus)
        if focus is None:
            focus = get_handle(alpha)
        c = torch.cat([h.squeeze(0), h_attn, focus], dim=1)
        y = self.output(c)
        return y, h, alpha, c

    def get_h(self, x, h_):
        z = self.emb(x.view(1, 1))
        _, h = self.rnn(z, h_)
        return h

    def pred_on_batch(self, imgs, sentences, lens):
        """
        input as a batch, for pure supervised learning, without focus
        """
        h, h_img, _ = self.get_initial_state(imgs)

        target_sz = [sentences.size(0)] + list(h_img.size())
        h_imgs = h_img.unsqueeze(0).expand(target_sz)
        h_imgs = pack_padded_sequence(h_imgs, lens).data
        s = var(torch.zeros(h_imgs.size(0), 3))

        z = self.emb(sentences)
        z = pack_padded_sequence(z, lens)
        hs, h = self.rnn(z, h)
        hs = hs.data
        h_attn = self._get_context(hs, h_imgs)[0]
        y = torch.cat([hs, h_attn, s], dim=1)
        return self.output(y)

    def _get_context(self, hs, h_imgs, focus=None):
        """
        Inputs: hs (_, hid_size),
                h_imgs (_, H, W, img_size)
                focus None or (1, 3)
        Outputs: h_attn (_, img_size), alpha (_, n)
        """

        isz = h_imgs.size(-1)
        hsz = hs.size(-1)
        n = h_imgs.size(1) * h_imgs.size(2)

        if self.attention == 'none' or type(focus) == torch.autograd.Variable:
            a = var(torch.zeros(h_imgs.size(0), n))
        elif self.attention == 'fc':
            target_sz = list(hs.size())
            target_sz.insert(1, n)
            hs_ = hs.unsqueeze(1).expand(target_sz).contiguous()
            # image information
            a = self.attn(torch.cat([hs_.view(-1, hsz),
                                     h_imgs.view(-1, isz)],
                                    dim=1)).view(-1, n)
        elif self.attention == 'dot':
            a = torch.bmm(h_imgs.view(-1, n, isz), hs.unsqueeze(-1))
            a = a.squeeze(2)

        if focus is not None:
            a = self._focus(a.view(h_imgs.size(1), h_imgs.size(2)),
                            focus.squeeze()).view(-1, n)

        if self.attention == 'none' and focus is None:
            size = (h_imgs.size(0), h_imgs.size(1) * h_imgs.size(2),
                    h_imgs.size(3))
            alpha = torch.exp(a).view(-1, n, 1)
            h_attn = alpha.expand(*size) * h_imgs.view(*size)
            h_attn = h_attn.max(1)[0]
        else:
            alpha = F.softmax(a, dim=1).view(-1, 1, n)
            h_attn = torch.bmm(alpha, h_imgs.view(-1, n, isz)).view(-1, isz)

        # output
        # return h_attn, alpha.squeeze(1) \
        #     .view(-1, h_imgs.size(1), h_imgs.size(2))
        return h_attn, alpha.view(-1, h_imgs.size(1), h_imgs.size(2))

    def _focus(self, a, focus):
        """
        Inputs: a (H, W),
                focus (3, )
        Outputs: a (1, H, W)
        """
        H = a.size(0)
        W = a.size(1)
        i_layer = var(torch.arange(H) / H).view(H, 1).expand_as(a)
        j_layer = var(torch.arange(W) / W).view(1, W).expand_as(a)
        x_layer = F.sigmoid(focus[0]).view(1, 1).expand_as(a)
        y_layer = F.sigmoid(focus[1]).view(1, 1).expand_as(a)
        sigma = F.sigmoid(focus[2]) ** 2 / 16 + 1e-6
        sigma_layer = sigma.view(1, 1).expand_as(a)
        a = -((i_layer - x_layer) ** 2 / 4 +
              (j_layer - y_layer) ** 2) / sigma_layer
        return a.unsqueeze(0)


class FF(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        return self.l2(x)


class Vision(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.model = None

    def forward(self, x):
        """
        Inputs: img (batch, channel, H, W)
        Outputs: fea (batch, out_dim, H // 8, W // 8)
        """
        x = 1. - x

        size = list(x.size())
        size[1] = 3
        x = x.expand(*size)

        y = F.tanh(self.model(x))
        return y


class SimpleVGG(Vision):
    def __init__(self, out_dim):
        super().__init__(out_dim)
        cfg = [8, 16, 'M', 32, 32, 'M', 64, out_dim, 'M']
        self.model = make_layers(cfg, batch_norm=True)


class SimpleResNet(Vision):
    def __init__(self):
        super().__init__(128)
        res = resnet18()
        self.model = nn.Sequential(*list(res.children())[:-4])


def get_handle(alpha):
    c = alpha.view(1, -1).max(1)[1].data[0]
    x = c % alpha.size(1) / alpha.size(1)
    y = c // alpha.size(1) / alpha.size(2)
    return var(torch.Tensor([[x, y, -2]]))


# from torchvision/models/vgg.py
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
