from __future__ import print_function
from tqdm import tqdm
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from demo_model import Dnn
from dataset.dnn_dataloader import MyCustomDataset, UniqueLabelsSampler


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=30, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=10200, help='input batch size')
parser.add_argument(
    '--num_classes', type=int, default=120, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='../weight', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
# parser.add_argument('--dataset', type=str, default='F:/ntu_120/train', required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='NTU120', help="dataset type NTU120|Kinetics400|RWF200")

opt = parser.parse_args()
print(opt)

# blue = lambda x: '\033[94m' + x + '\033[0m'
# fix seed
opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# dataset, dataloader, num_classes
train_dataset = MyCustomDataset(root_dir='F:/ntu_120/train', transform=None)
sampler = UniqueLabelsSampler(train_dataset, batch_size=opt.batchSize, num_classes=opt.num_classes)
# TODO:test dataset
test_dataset = MyCustomDataset(root_dir='F:/ntu_120/val', transform=None)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=sampler,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))
# print(len(train_dataset), len(test_dataset))
num_classes = len(train_dataset.classes)
# print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if __name__ == '__main__':
    classifier = Dnn(k=num_classes)
    # 检查是否有可用的CUDA设备
    if torch.cuda.is_available():
        device = torch.device('cuda')  # 设备对象表示CUDA设备
        classifier = classifier.to(device)  # 将模型转移到CUDA设备上
        for layer in classifier.modules():  # 遍历模型的所有层
            if isinstance(layer, torch.nn.ReLU):  # 检查是否为激活函数层
                layer.cuda()
    else:
        device = torch.device('cpu')  # 设备对象表示CPU

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_batch = len(train_dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        scheduler.step()
        for i, data in enumerate(train_dataloader, 0):
            points, target = data
            # 确保输入数据也在相同的设备上
            # points = points.to(device)  # 将输入数据转移到与模型相同的设备上
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)

            loss = F.cross_entropy(pred, target)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    # validation
    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(test_dataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))
