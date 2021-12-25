#encoding=utf-8
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from torch.autograd import Function
import time
import pdb


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    return GRLayer.apply(x)


class Img_G(nn.Module):
    def __init__(self):
        super(Img_G, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x=self.conv(x)
        # pdb.set_trace()
        return x


class Img_D(nn.Module):
    def __init__(self):
        super(Img_D, self).__init__()

        self.sigmoid = nn.Sigmoid()

        model = [nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.sigmoid(self.model(x)).squeeze(1)
        return x


class Ins_D(nn.Module):
    def __init__(self):
        super(Ins_D, self).__init__()
        self.model = nn.Linear(4096, 2)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.model(x)
        return x



class Img_D_res(nn.Module):
    def __init__(self):
        super(Img_D_res, self).__init__()

        self.sigmoid = nn.Sigmoid()

        model = [   nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                            nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                            nn.InstanceNorm2d(256),
                            nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.sigmoid(self.model(x)).squeeze(1)
        return x


class Ins_D_res(nn.Module):
    def __init__(self):
        super(Ins_D_res, self).__init__()
        self.model = nn.Linear(2048, 2)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.model(x)
        return x