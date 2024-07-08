from setuptools import setup, find_packages

setup(
    name="wormdynamics",
    version="0.1",
    packages=find_packages(),
    author="Alicia Lu",
    author_email="alicialu@mit.edu",
    install_requires=["numpy", "torch"],
    license="MIT",
    description="transformer to predict missing neuronal activities in C. elegans whole-brain GCaMP recordings",
    url="https://github.com/aletheia88/worm-transformer/"
)
