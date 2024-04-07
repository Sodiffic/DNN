from transformers import (
    CLIPTextConfig,
    CLIPVisionConfig,
    BertConfig,
    CLIPConfig
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

'''
这部分是完整预训练对比学习用到的configuration。目前已有bert部分，待修改dnn分类训练部分。
bert-clip-config 通过引用clipconfig来定义本项目中需要的。
待修改：将dnn的config形式与clip的方式对齐。。。
'''

class BertCLIPConfig(CLIPConfig):
    r"""
    [`CLIPConfig`] is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate
    CLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    Example:
    ```python
    from transformers import CLIPConfig, CLIPModel
    # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
    configuration = CLIPConfig()
    # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    model = CLIPModel(configuration)
    # Accessing the model configuration
    configuration = model.config
    # We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
    # Initializing a CLIPText and CLIPVision configuration
    config_text = CLIPTextConfig()
    config_vision = CLIPVisionConfig()
    config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "clip"
    is_composition = True

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ): # 初始化：dim:512
        super().__init__(**kwargs)

        # 如果存在`_config_dict`，我们使用它们来实现向后兼容性。
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        if text_config_dict is not None:
            text_config = text_config_dict
        if vision_config_dict is not None:
            vision_config = vision_config_dict

        # `_config_dict`不存在时使用默认初始化。
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the CLIPTextConfig with default values.")
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the CLIPVisionConfig with default values.")

        # text_config用bertconfig初始化，vision_config用CLIPVisionConfig初始化，同时初始化projection_dim，logit_scale_init_value，initializer_factor
        # TODO: ！！vision_config = CLIPVisionConfig(**vision_config)重写config
        self.text_config = BertConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

