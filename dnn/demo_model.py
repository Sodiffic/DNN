from __future__ import print_function
import torch
import PIL
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bottleneck_rate=0.25):
        super(BasicBlock, self).__init__()
        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        self.mid_channel = int(out_channel * bottleneck_rate)
        # 三层mlp，
        self.conv1 = torch.nn.Conv1d(self.in_channel, self.mid_channel, 1)
        self.conv2 = torch.nn.Conv1d(self.mid_channel, self.mid_channel, 1)
        self.conv3 = torch.nn.Conv1d(self.mid_channel, self.out_channel, 1)

        self.bn1 = nn.BatchNorm1d(self.mid_channel)
        self.bn2 = nn.BatchNorm1d(self.out_channel)

        if in_channel == out_channel:
            self.shortcut_conv = nn.Identity()
        else:
            self.shortcut_conv = nn.Conv1d(self.in_channel, self.out_channel, 1)

    def forward(self, x):
        shortcut = x
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn1(self.conv2(x)))
        # print(x.shape)
        x = self.bn2(self.conv3(x))
        # print(x.shape)

        return F.relu(x + self.shortcut_conv(shortcut))


class Dnn(nn.Module):
    def __init__(self, k=2, stage=[1,1,1,1]):
        super(Dnn, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(7, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.get_feature_list = []
        in_chan = 16
        out_chan = 64
        for i in range(len(stage)):  # 每个stage内的block数
            for j in range(0, stage[i]):
                """
                stage1: [32->32->32->128][128->32->32->128]
                stage2: [128->64->64->256][256->64->64->256]
                stage3: [256->128->128->512][512->128->128->512]
                stage4: [512->256->256->1024][1024->256->256->1024]
                """
                self.get_feature_list.append(BasicBlock(in_channel=in_chan, out_channel=out_chan, bottleneck_rate=0.25))
                in_chan = out_chan
                out_chan = out_chan if (j==0 and stage[i]>1) else out_chan*2

        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k)
        # TODO:待修改
        self.get_feature_list = nn.ModuleList(self.get_feature_list)

    def forward(self, x):
        # 第一层mlp
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, n_pts, 7)--> [30, 32, 10200]
        # print(x.shape)
        # residual mlp blocks
        i = 0
        for get_feature in self.get_feature_list:
            x = get_feature(x)
            i += 1
            # print(i,':', x.shape, '\n')

        # Maxpooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)  # 调整形状 (batch_size,1024)
        # print(x.shape)
        # TODO:还差一个classification。。。下午来写

        x = F.relu(self.bn2(self.fc1(x)))  # (batch_size,512)
        x = F.relu(self.bn3(self.dropout(self.fc2(x))))  # (batch_size,256)
        x = self.fc3(x)  # (batch_size,k)
        # 返回的是该点云是第ki类的对数概率分布
        # print('k的大小：',self.k, '全连接后的大小：',x.size())
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    sim_data = Variable(torch.rand(30, 7, 10200))  # [batchsize, channel, point_num]
    trans = Dnn()
    out = trans(sim_data)
    print('mlp', out.size())