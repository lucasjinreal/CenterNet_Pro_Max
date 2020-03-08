from alfred.utils.log import logger as logging

from models.centernet import CenterNet
from models.backbone import Backbone
from models.ops import ShapeSpec
from models.backbone import ResnetBackbone
from models.head import CenternetDeconv
from models.head import CenternetHead
from models.centernet import CenterNet


from configs.centernet_config import config


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = ResnetBackbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_upsample_layers(cfg, ):
    upsample = CenternetDeconv(cfg)
    return upsample


def build_head(cfg, ):
    head = CenternetHead(cfg)
    return head


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_upsample_layers = build_upsample_layers
    cfg.build_head = build_head
    model = CenterNet(cfg)
    return model


def train():
    model = build_model(config)
    logging.info('model initialized.')


if __name__ == '__main__':
    train()
