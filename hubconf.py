dependencies = ['torch', 'torchvision', 'supervision', 'timm', 'clip', 'detectron2']


def _install():
    import os
    import sys
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', os.path.dirname(__file__)], stdout=sys.stdout, stderr=sys.stderr)


def detic(vocab='lvis', **kw):
    """Detic - open-vocabulary object detector
    """
    _install()
    from detic import Detic
    model = Detic(vocab, **kw).eval()
    return model
