from setuptools import setup, find_packages

setup(
    name="mazu_saudi_ewai",
    version="0.1.0",
    description="MAZU 多灾种早期预警智能体 - 沙特阿拉伯目标区域",
    author="MAZU Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "xarray>=2023.1.0",
        "netCDF4>=1.6.0",
        "lightgbm>=4.0.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "networkx>=3.0",
        "gradio>=4.0.0",
        "matplotlib>=3.7.0",
        "cartopy>=0.21.0",
        "tqdm>=4.65.0",
    ],
)
