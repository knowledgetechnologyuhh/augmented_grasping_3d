import setuptools
from setuptools.extension import Extension
import numpy as np

with open('requirements.txt') as f:
    requirements = f.read().splitlines()



setuptools.setup(
    name             = 'keras-transformer',
    version          = '0.1.0',
    description      = 'Keras implementation of the Transformer Network for sequence to sequence translation. '
                       'This is a modification of https://github.com/Lsdefine/attention-is-all-you-need-keras/',
    url              = 'https://github.com/knowledgetechnologyUHH',
    author           = 'Fares Abawi',
    author_email     = 'abawi@informatik.uni-hamburg.de',
    maintainer       = 'Fares Abawi',
    maintainer_email = 'abawi@informatik.uni-hamburg.de',
    packages         = setuptools.find_packages(),
    install_requires = requirements,
    entry_points     = {
        'console_scripts': [
            'transformer-train=keras_transformer.bin.train:main'
        ],
    }
)
