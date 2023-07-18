import setuptools

setuptools.setup(
    name='detic',
    version='0.0.1',
    description='Detecting Twenty-thousand Classes using Image-level Supervision',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    # packages=['detic', 'centernet'],
    # package_dir={'detic': 'detic', 'centernet': 'third_party/CenterNet2/centernet'},
    install_requires=[
        'torch', 'torchvision', 'supervision', 'timm',
        'clip @ git+https://github.com/openai/CLIP.git@main#egg=clip',
        'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@main#egg=detectron2',
    ],
    extras_require={})
