from setuptools import setup, find_packages

import ppacls

VERSION = ppacls.__version__


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_requirements():
    with open('./requirements.txt', encoding="utf-8") as f:
        requirements = f.readlines()
    return requirements


if __name__ == "__main__":
    setup(
        name='ppacls',
        packages=find_packages(),
        author='yeyupiaoling',
        version=VERSION,
        install_requires=parse_requirements(),
        description='Audio Classification toolkit on PaddlePaddle',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='https://github.com/yeyupiaoling/AudioClassification_PaddlePaddle',
        download_url='https://github.com/yeyupiaoling/AudioClassification_PaddlePaddle.git',
        keywords=['audio', 'paddle'],
        classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Natural Language :: Chinese (Simplified)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9', 'Topic :: Utilities'
        ],
        license='Apache License 2.0',
        ext_modules=[])
