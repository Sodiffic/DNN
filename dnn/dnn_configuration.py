from transformers import PretrainedConfig
from typing import List
'''
用自己的方法改写dnn对应的config，待完成对接
'''
class ResnetConfig(PretrainedConfig):
    model_type = "resmlp"
    # model_type = "resnet"

    #TODO: 对照pointnet找到对应参数
    def __init__(
        self,
        block_type= "bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,  # RGB:3
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block` must be 'basic' or bottleneck', got {block}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {block}.")

        self.block_type = block_type  # Basic block是ResNet的基本构建块，而Bottleneck block是ResNet较深层网络（如ResNet-50, ResNet-101等）中使用的更复杂的构建块，它包含了更多的卷积层
        self.layers = layers  # 每个阶段stage的卷积层数,eg resnet50:[3, 4, 6, 3]
        self.num_classes = num_classes  # 网络的输出层应该有多少个神经元，通常对应于数据集中的类别数
        self.input_channels = input_channels  # 指定输入图像的通道数。RGB:3
        self.cardinality = cardinality  # 在ResNeXt中，这个参数指定了分组卷积中每个组的数量。在原始ResNet中，这个值通常设为1，表示不使用分组卷积。
        self.base_width = base_width  # 用于设置构建块的宽度（即每个卷积层的通道数）。??
        self.stem_width = stem_width  # 定网络初始部分（stem）的输出通道数。??
        self.stem_type = stem_type  # 字符串类型，指定stem的类型。这里支持空字符串（表示使用默认的stem结构）、"deep"和"deep-tiered"。
        self.avg_down = avg_down  #??
        super().__init__(**kwargs)
