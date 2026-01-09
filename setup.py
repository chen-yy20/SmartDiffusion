from setuptools import setup, find_packages

setup(
    name="smart-diffusion",
    version="0.1",
    packages=find_packages(),
    install_requires=[
    ],
    package_dir={
        'chitu_core': 'chitu_core',
        'chitu_diffusion': 'chitu_diffusion'
    }
)