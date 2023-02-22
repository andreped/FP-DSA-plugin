import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastpathology",
    version="0.0.1",
    author="André Pedersen",
    author_email="andrped94@gmail.com",
    description="Package for seemlessly using FAST pipelines in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andreped/FP-dsa-plugin/tree/main/dsa/fastpathology",
    packages=setuptools.find_packages(),
    install_requires=[
        "pyFAST",
    ],
    entry_points={
        'console_scripts': [
            'fastpathology = fastpathology.deploy:main',
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.6",
)
