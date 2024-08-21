from torch import nn
import math
import torch
import torch.nn as nn

class MultDownscaleConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(MultDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor

    def forward(self, y, lr):
        y = y.clone()
        out = self.pool(y)
        out = y * torch.kron(lr * 1 / out, torch.ones((self.upsampling_factor, self.upsampling_factor)).to('cuda'))
        return out

class AddDownscaleConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(AddDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor

    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        out = y + torch.kron(lr - sum_y, torch.ones((self.upsampling_factor, self.upsampling_factor)).to('cuda'))
        return out


class ScAddDownscaleConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(ScAddDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor

    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        diff_P_x = torch.kron(lr - sum_y, torch.ones((self.upsampling_factor, self.upsampling_factor)).to('cuda'))
        sigma = torch.sign(-diff_P_x)
        out = y + diff_P_x * (sigma + y) / (
                    sigma + torch.kron(sum_y, torch.ones((self.upsampling_factor, self.upsampling_factor)).to('cuda')))
        return out


class SoftmaxConstraints(nn.Module):
    def __init__(self, upsampling_factor, exp_factor=1):
        super(SoftmaxConstraints, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)

    def forward(self, y, lr):
        y = torch.exp(y)
        sum_y = self.pool(y)
        out = y * torch.kron(lr * 1 / sum_y, torch.ones((self.upsampling_factor, self.upsampling_factor)).to('cuda'))
        return out


class GlobalConstraints(nn.Module):
    """ Layer constraining the Generator output to conserve the sum of the input feature values.
        Before rescaling the output, inverse transforms are applied to convert the input and output
        to precipitation units.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y, lr):
        norm_fraction = lr.sum(dim=(2, 3)) / y.sum(dim=(2, 3))
        norm_fraction = norm_fraction.unsqueeze(-1)
        norm_fraction = norm_fraction.unsqueeze(-1)
        out = y * norm_fraction * 9

        return out