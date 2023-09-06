dependencies = [
    'torch', #'torchvision', 'supervision', 'timm', 'clip', 'detectron2'
]


def _install():
    import os
    import sys
    import subprocess
    # import importlib.util
    # if importlib.util.find_spec('detic') is None:
    try:
        import detic
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', os.path.dirname(__file__)])


def detic(vocab='lvis', **kw):
    """Detic - open-vocabulary object detector
    """
    _install()
    from detic import Detic
    model = Detic(vocab, **kw).eval()
    return model
