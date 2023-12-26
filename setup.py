from setuptools import setup, find_packages

setup(
    name='Detic-hachix',
    version='1.0.0',
    description='Detic: Detecting Twenty-thousand Classes using Image-level Supervision',
    author='Facebook AI Research',
    url='https://github.com/HACHIX-CORPORATION/Detic',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'mss',
        'timm',
        'dataclasses; python_version<"3.7"',  # dataclasses is a standard library in Python 3.7 and later
        'ftfy',
        'regex',
        'fasttext-wheel',
        'scikit-learn',
        'lvis',
        'nltk'
    ],
    dependency_links=[
        'git+https://github.com/openai/CLIP.git#egg=clip'
    ],
)
