from setuptools import setup, find_packages

setup(
    name="mws",  # 包名
    version="0.1",
    author="maxi",
    author_email="mx1471345528@163.com",
    description="A simple Chinese Mult-Word Segment",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mws",  # 你的 GitHub 项目地址
    packages=find_packages(),
    install_requires=[
        # 列出依赖的库，例如：'numpy>=1.18.0'，如果没有依赖可以留空
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  # 最低 Python 版本要求
)
