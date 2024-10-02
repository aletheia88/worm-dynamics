from setuptools import setup, find_packages

setup(
    name="attention_predict",
    version="1.0",
    packages=find_packages(),
    install_requires=["numpy", "torch"],
    license="MIT"
)
