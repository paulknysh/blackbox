import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="black_box",
    version="1.0.2",
    author="Paul Knysh",
    author_email="paul.knysh@gmail.com",
    description="A Python module for parallel optimization of expensive black-box functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paulknysh/blackbox",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
