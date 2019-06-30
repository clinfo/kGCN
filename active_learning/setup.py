import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="active_learing",
    version="1.0.0",
    author="Taro Kiritani",
    author_email="taro.kiritani@exwzd.com",
    packages=setuptools.find_packages(),
    # rdkit needs to be installed separately because it is not listed on pypi.
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6"
    ],
)
