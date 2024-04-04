import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
"""
自定义demo版NTU120的dataset和sampler，实现获取点集的功能
其中train和validate共用一个dataset，sampler用于训练时平衡类别
"""


# TODO:支持其他数据集；数据增强
class MyCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 文件地址
        self.transform = transform
        self.samples = []  # (data, label)对
        self.classes = {}  # (label,[data])
        """
        类别处理:形成index-classname-很多sample的映射关系        
        """
        sample_file = glob.glob(f'{root_dir}/*.npy', recursive=True)
        for path in sample_file:
            _, filename = os.path.split(path)
            cat = filename.split('_')[0]  # 类别名
            # print(cat)
            self.samples.append((path, cat))
            # self.classes[cat] = []
            # self.classes[cat].append(path)
        # print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 根据index返回一个样本和对应标签的张量
        # print(idx.type())
        # path, cls = self.samples[100]
        path, cls = self.samples[idx]
        points = np.load(path).reshape(-1, 7)  # 读取文件，转成张量
        points = torch.from_numpy(points).float()
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))  # 整数标签转成张量

        if self.transform:
            points = self.transform(points)
        return points, cls  # 返回图像和类别标签（这里假设类别是字符串形式）


class UniqueLabelsSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size, num_classes):
        self.num_classes = num_classes
        self.data_source = data_source
        self.batch_size = batch_size
        self.labels = [int(item[1]) for item in data_source.samples]  # sample是一个(data, label)对的列表
        # TODO:这个labels究竟是什么？
        self.original_indices = {label: [] for label in set(self.labels)}
        # print(len(self.original_indices))
        for idx, label in enumerate(self.labels):
            self.original_indices[label].append(idx)
        self.label_to_indices = {label: list(indices) for label, indices in self.original_indices.items()}
        # 打乱每个类别内部的索引顺序
        for label in self.label_to_indices:
            self.label_to_indices[label] = torch.randperm(len(self.label_to_indices[label])).tolist()

        # 计算总共可以组成多少个完整的batch
        self.num_samples = min(len(indices) for indices in self.label_to_indices.values()) * self.batch_size
        print('总共可以组成的完整的batch', self.num_samples)

    #
    def reset_label_to_indices(self):
        """
        在每个epoch开始时重置label_to_indices字典并对每一个标签里面的元素打乱。
        """
        # TODO:!
        self.label_to_indices = {label: list(indices) for label, indices in self.original_indices.items()}
        for item in self.label_to_indices.items():
            random.shuffle(item[1])

    def __iter__(self):
        self.reset_label_to_indices()  # 重置
        indices = []
        for i in range(self.num_samples // self.batch_size):
            batch_indices = []
            # 为当前batch选择不同类别的样本
            selected_labels = torch.randperm(self.num_classes)[:self.batch_size].tolist()  # 随机选择类别
            for label in selected_labels:
                if len(self.label_to_indices[label]) > 0:
                    idx = self.label_to_indices[label].pop(0)  # 弹出当前类别的一个样本索引
                    batch_indices.append(idx)

            # 如果批次中的样本数量少于batch_size，则从剩余类别中随机选择更多样本
            while len(batch_indices) < self.batch_size:
                remaining_labels = [l for l in self.label_to_indices if len(self.label_to_indices[l]) > 0]
                if not remaining_labels:
                    break  # 如果所有类别的样本都用完了，就退出循环
                label = torch.choice(remaining_labels)  # 随机选择一个还有剩余样本的类别
                idx = self.label_to_indices[label].pop(0)
                batch_indices.append(idx)
                # for label in self.label_to_indices:
            #     if len(self.label_to_indices[label]) > 0:
            #         idx = self.label_to_indices[label].pop(0)  # 弹出当前类别的一个样本索引
            #         batch_indices.append(idx)
            indices.extend(batch_indices)
            # 如果有需要，可以在这里打乱当前batch内部的顺序
            # torch.randperm(self.batch_size)
        print('indices：', list(indices))
        return iter(indices)

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':

    # 创建数据集实例 F:/ntu_120/train
    dataset = MyCustomDataset(root_dir='F:/ntu_120/train', transform=None)
    # print(len(dataset))
    ps, cls = dataset[0]
    print(ps.size(), cls.size())

    # 定义batch size
    batch_size = 30
    # 创建sampler实例
    sampler = UniqueLabelsSampler(dataset, batch_size, num_classes=120)
    # 创建DataLoader实例
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)  #
    # for idx in sampler:
    #     print(idx, dataset[idx][1])

    # test: 遍历数据加载器，检查索引是否正确
    for epoch in range(1):  # 只测试一个epoch
        # print(list(dataloader))
        # for idx, (data, target) in enumerate(dataloader):
        #     print(f"Batch {idx + 1}:")
        #     print("Data shape:", data.shape)
        #     print("Target type:", type(target))
        #     print("Target shape:", target.shape)
        #     print("Target content:", target)

        for batch_indices in dataloader:
            # print(batch_indices[1])
            # 验证索引的数量是否正确
            # assert len(batch_indices) == batch_size, "Batch size does not match"
            # 使用索引从数据集中提取数据和标签
            """这里的idx是tensor，正式使用的时候需要注意"""
            batch_data, batch_labels = batch_indices
            # print(batch_labels[:5])

            # 检查数据和标签的形状等属性
            # 这里您可以添加更多的断言来验证数据和标签的正确性
            assert len(batch_data) == batch_size, "Batch data size does not match"
            assert len(batch_labels) == batch_size, "Batch labels size does not match"

            # 输出批次数据和标签，用于检查（可选）
            print(batch_data[:5])  # 打印前5个数据样本
            print(batch_labels[:5])  # 打印前5个标签

            # 可以在这里添加更多测试逻辑，比如检查每个类别的样本是否均匀分布等

    # 如果上面的代码没有抛出异常，并且输出看起来合理，那么您的数据加载器可能工作正常
    print('success')
