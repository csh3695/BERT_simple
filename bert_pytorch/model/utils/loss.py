import torch
import torch.nn as nn


class FSPLoss(nn.Module):
    def __init__(self, t_layer=12, s_layer=6, stride=2):
        super(FSPLoss, self).__init__()
        self.t_layer = t_layer
        self.s_layer = s_layer
        self.stride = stride
        mask = torch.zeros(self.s_layer, self.s_layer)
        for i in range(self.s_layer-1):
            mask[i, i+1:] = 1
        self.register_buffer('_tri_mask', mask.bool())
        self.lmda = 2
        self.loss = nn.MSELoss()

    def forward(self, y, x):
        # x: 12*b*512*512
        # y: 6*b*512*512
        x = x[list(range(0, self.t_layer, self.stride))]
        assert(x.shape == y.shape)
        x = x.reshape(self.s_layer, -1)
        y = y.reshape(self.s_layer, -1)
        g_x = (torch.matmul(x, x.transpose(-1, -2))/x.shape[-1])[self._tri_mask]
        g_y = (torch.matmul(y, y.transpose(-1, -2))/y.shape[-1])[self._tri_mask]
        return self.lmda * self.loss(g_y, g_x)


class SoftLabelLoss(nn.Module):
    def __init__(self, t=10, lmda=20):
        super(SoftLabelLoss, self).__init__()
        self.smax = nn.Softmax(dim=-1)
        self.t = t
        self.lmda = lmda

    def forward(self, x, y):
        return - self.lmda * torch.mul(self.smax(y/self.t), (self.smax(x/self.t)+(1e-7)).log2()).mean()
