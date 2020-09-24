import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()



setuptools.setup(
    name             = 'keras-fusionnet',
    version          = '0.1.0',
    description      = 'Keras implementation of Fusion Network for combining image and textual data',
    url              = 'https://github.com/knowledgetechnologyUHH',
    author           = 'Fares Abawi',
    author_email     = 'abawi@informatik.uni-hamburg.de',
    maintainer       = 'Fares Abawi',
    maintainer_email = 'abawi@informatik.uni-hamburg.de',
    packages         = setuptools.find_packages(),
    install_requires = requirements,
    entry_points     = {
        'console_scripts': [
            'fusionnet-train=keras_fusionnet.bin.train:main'
        ],
    }
)
