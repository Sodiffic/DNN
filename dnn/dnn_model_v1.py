import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import timm
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, LayerType, create_attn, \
    get_attn, get_act_layer, get_norm_layer, create_classifier
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.layers import LayerType, create_classifier
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations

"""
      stage1: [32->32->32->128][128->32->32->128]
      stage2: [128->64->64->256][256->64->64->256]
      stage3: [256->128->128->512][512->128->128->512]
      stage4: [512->256->256->1024][1024->256->256->1024]
      pooling-full_connect:
      1024-512-256-k
"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: 输入通道维度
            planes: 用于确定输出通道尺寸
            stride: 卷积层中使用的步幅。
            downsample: 残差路径的可选下采样层。可能需要使用max-pooling
            cardinality: 卷积组的个数。基础的resnet=1
            base_width: 用于确定输出通道维度的基本宽度。（待确定）
            reduce_first: 残差块的第一卷积输出宽度的缩减因子。原始resnet不改变。常见0.5使用1x1卷积将通道减少一半。
            dilation: 卷积层的膨胀（空洞）率。因为mlp里都是conv1d所以这里不用考虑。
            first_dilation: 第一卷积层的膨胀率，因为mlp里都是conv1d所以这里不用考虑。
            act_layer: 激活层
            norm_layer: 归一化层
            attn_layer: 注意力层（待删）
            aa_layer: 抗锯齿层（待删）
            drop_block: DropBlock层的类（待删）
            drop_path: 可选DropPath层（待删）
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion  # 计算卷积层输出通道的数量
        first_dilation = first_dilation or dilation  # 如果first_dilation是一个非假值，那么保持其原值；否则，first_dilation被赋值为dilation。
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)  # 是否应该使用抗锯齿（anti-aliasing，简称aa）层

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x

class Bottleneck(nn.Module):
    r"""basic bottleneck?"""
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 0,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm1d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        # use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv1d(inplanes, first_planes)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        # self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        # self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        # self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # block1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # block2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        # block3
        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    r"""我需要用这个下采样改变输入的权重。其实对我来说只是一个conv1d（？是否需要？）"""
    norm_layer = norm_layer or nn.BatchNorm1d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[nn.Conv1d(in_channels, out_channels), norm_layer(out_channels)])  # 很怪 这里不做


def downsample_avg(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    r"""和上一个比起来只是多了一步pooling。我可能不需要这个。"""
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])

# 用于将每一个stage里的block连接起来
def make_blocks(
        block_fn: Union[BasicBlock, Bottleneck],
        channels: List[int],
        block_repeats: List[int],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        drop_block_rate: float = 0.,
        drop_path_rate: float = 0.,
        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1

    # 构建resnet的stage 以resnet50的四阶段[3,4,6,4]为例，
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2  # 对stride处理。。。我觉得我不需要这个stride
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:  # 这里符合我的设想：如果输入和输出通道不一样，那么就要设计下采样
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None  #
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet(nn.Module):

    def __init__(
            self,
            block: Union[BasicBlock, Bottleneck],  # BasicBlock和 Bottleneck
            layers: List[int],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            global_pool: str = 'max',
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 64,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            avg_down: bool = False,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.,
            drop_block_rate: float = 0.,
            zero_init_last: bool = True,
            block_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            block (nn.Module): 残差块的类。选项有BasicBlock、Bottleneck。
            layers (List[int]) : 每块里的层数
            num_classes (int): 分类类别数 (default 1000)
            in_chans (int): 输入通道数 (default 3)
            output_stride (int): 网络的输出stride， 32, 16, or 8. (default 32)
            global_pool (str): 全局池化方法， 'avg', 'max', 'avgmax', 'catavgmax' 中的一种(default 'avg')
            cardinality (int): 3x3卷积的卷积组数量 (default 1)
            base_width (int): 卷积通道因子 `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): stem convolutions通道数 (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem部分
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64  # input:
        # if deep_stem:
        #     stem_chs = (stem_width, stem_width)
        #     if 'tiered' in stem_type:
        #         stem_chs = (3 * (stem_width // 4), stem_width)
        #     self.conv1 = nn.Sequential(*[
        #         nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
        #         norm_layer(stem_chs[0]),
        #         act_layer(inplace=True),
        #         nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
        #         norm_layer(stem_chs[1]),
        #         act_layer(inplace=True),
        #         nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        # else:
        #     self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        """stem部分这么改了，改成一层mlp"""
        self.conv1 = nn.Conv1d(in_chans, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
        self.maxpool = nn.MaxPool1d(1) # 不确定这么写对不对
        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        # if replace_stem_pool:
        #     self.maxpool = nn.Sequential(*filter(None, [
        #         nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
        #         create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
        #         norm_layer(inplanes),
        #         act_layer(inplace=True),
        #     ]))
        # else:
        #     if aa_layer is not None:
        #         if issubclass(aa_layer, nn.AvgPool2d):
        #             self.maxpool = aa_layer(2)
        #         else:
        #             self.maxpool = nn.Sequential(*[
        #                 nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #                 aa_layer(channels=inplanes, stride=2)])
        #     else:
        #         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [32, 64, 128, 256]  # 通道数
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True):
        for n, m in self.named_modules():
            # 对relu进行初始化
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # stem部分：卷积+batchNorm+relu+max_pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            # 进行梯度检查
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            # 不进行梯度检查
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """两部分：池化+全连接"""
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""最主要的部分，一个是forward_feature, 一个是forward_head"""
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def _create_resnet(variant, pretrained: bool = False, **kwargs) -> ResNet:
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


@register_model
def resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
    return _create_resnet('resnet50', pretrained, **dict(model_args, **kwargs))
