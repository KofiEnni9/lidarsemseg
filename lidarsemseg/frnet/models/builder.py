
from mmengine.registry import Registry

MODELS = Registry('model')
DATASETS = Registry('dataset')
TRANSFORMS = Registry('transform')


def build_model(cfg):
    """Build models."""
    return MODELS.build(cfg)