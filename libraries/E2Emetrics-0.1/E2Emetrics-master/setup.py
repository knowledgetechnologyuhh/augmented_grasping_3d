import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

VERSION = '0.1.0'

setuptools.setup(
    name='e2emetrics',
    version=VERSION,
    author='Xinlei Chen',
    url='https://github.com/code-lava/e2e-metrics',
    description='E2E metrics include coco tools for evaluation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Plugins',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows', # XP
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: BLEU :: METEOR :: CIDER :: ROUGE :: COCO :: '
        'NLP :: Natural Language Generation :: Machine Translation :: Evaluation Tools',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    data_files=[('e2emetrics/mteval', ['e2emetrics/mteval/mteval-v13a-sig.pl']), 
		('e2emetrics/pycocoevalcap/meteor', ['e2emetrics/pycocoevalcap/meteor/meteor-1.5.jar']),
		('e2emetrics/pycocoevalcap/meteor/data', ['e2emetrics/pycocoevalcap/meteor/data/paraphrase-en.gz']),
		('e2emetrics/pycocoevalcap/tokenizer', ['e2emetrics/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar'])],
    packages= setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
