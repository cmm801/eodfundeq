from setuptools import setup, find_packages

setup(
    name='eodfundeq',
    version='0.0.1',
    author='Christopher Miller',
    author_email='cmm801@gmail.com',
    packages=find_packages(), 
    include_package_data=True,
    scripts=[],
    url='http://pypi.python.org/pypi/eodfundeq/',
    license='MIT',
    description='A package for downloading data from eodhistoricaldata.com.',
    long_description=open('README.md').read(),
    install_requires=[
        'eodhistdata',
        'numpy',
        'pandas',
        'setuptools-git',
    ],
)
