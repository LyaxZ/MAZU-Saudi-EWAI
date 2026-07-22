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
        "h5netcdf>=1.0.0",
        "scipy>=1.10.0",
        "lightgbm>=4.0.0",
        "scikit-learn>=1.2.0",
        "shap>=0.41.0",
        "networkx>=3.0",
        "openai>=1.55.0",
        "python-dotenv>=1.0.0",
        "gradio>=4.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "mazu-web = app.gradio_app:main",
            "mazu-cli = app.chat_cli:main",
        ],
    },
)
