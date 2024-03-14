import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvBNReLU1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.Relu(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# TODO:处理一下in_channel 和 out_channel
class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True):
        super(ConvBNReLURes1D, self).__init__()
        self.act = nn.Relu(inplace=True)
        # self.net1 = nn.Sequential(
        #     nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
        #               kernel_size=kernel_size, groups=groups, bias=bias),
        #     nn.BatchNorm1d(int(channel * res_expansion)),
        #     self.act
        # )
        # if groups > 1:
        #     self.net2 = nn.Sequential(
        #         nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
        #                   kernel_size=kernel_size, groups=groups, bias=bias),
        #         nn.BatchNorm1d(channel),
        #         self.act,
        #         nn.Conv1d(in_channels=channel, out_channels=channel,
        #                   kernel_size=kernel_size, bias=bias),
        #         nn.BatchNorm1d(channel),
        #     )
        # else:
        #     self.net2 = nn.Sequential(
        #         nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
        #                   kernel_size=kernel_size, bias=bias),
        #         nn.BatchNorm1d(channel)
        #     )
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )




    def forward(self, x):
        if x.res_expansion == 1:
            self.act(self.net1(self.net1(self.net1(x)))+x)
        else

        # return self.act(self.net2(self.net1(x)) + x)


