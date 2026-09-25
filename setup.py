from setuptools import find_packages, setup


setup(
    name="signgpt",
    version="0.1.0",
    description="Gloss-free sign-language translation and generation",
    packages=find_packages(exclude=("configs", "data", "deps")),
    python_requires=">=3.9",
    install_requires=["numpy", "omegaconf", "torch", "tqdm"],
)
