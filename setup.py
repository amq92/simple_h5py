from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='simple_h5py',
    version='0.1.1',
    author='Arturo Mendoza',
    description='A simple wrapper for the h5py library.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/amq92/simple_h5py',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'h5py'],
    license='GNU Affero General Public License v3',
)
