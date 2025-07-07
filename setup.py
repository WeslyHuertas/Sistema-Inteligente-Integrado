"""Package installer"""

from setuptools import setup, find_packages

setup(
    name="demanda_streamlit_app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
    ],
)
