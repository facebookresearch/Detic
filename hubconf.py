dependencies = [
    'torch', 'torchvision', 'supervision', 'timm',
    'clip @ git+https://github.com/openai/CLIP.git@main#egg=clip',
    'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@main#egg=detectron2',
]


def detic(vocab='lvis', **kw):
    """Detic - open-vocabulary object detector
    """
    from detic import Detic
    model = Detic(vocab, **kw).eval()
    return model
