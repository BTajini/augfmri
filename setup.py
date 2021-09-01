from setuptools import setup, find_packages

setup(
    name="condica",
    description="Conditional ICA",
    version="0.0.0",
    keywords="",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=['numpy>=1.12', 'scikit-learn>=0.23', 'nilearn', 'joblib']
)
