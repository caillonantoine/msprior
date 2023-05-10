import os

import setuptools

version = os.environ["MSPRIOR_VERSION"]

with open("README.md", "r") as readme:
    readme = readme.read()

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

setuptools.setup(
    name="acids-msprior",
    version=version,
    author="Antoine CAILLON",
    author_email="caillon@ircam.fr",
    description=
    "MSPRIOR: A multiscale prior model for realtime temporal learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        'msprior/configs': ['*.gin'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "msprior = msprior_scripts.main_cli:main",
        ]
    },
    install_requires=requirements.split("\n"),
    python_requires='>=3.9',
    include_package_data=True,
)
