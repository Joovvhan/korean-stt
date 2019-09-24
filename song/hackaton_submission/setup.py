#nsml: nvcr.io/nvidia/pytorch:19.06-py3
from distutils.core import setup
import setuptools

setup(
    name='lilililill',
    version='1.0',
    install_requires=[
        'python-Levenshtein',
        'jamotools',
        'librosa'
    ]
)
