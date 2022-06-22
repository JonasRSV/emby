import setuptools

setuptools.setup(
    name="emby",
    version="0.0.1",
    author="Jonas Valfridsson",
    author_email="jonas@valfridsson.net",
    description="",
    url="https://github.com/JonasRSV/emby",
    packages=setuptools.find_packages(),
    install_requires=["numba==0.49.0", "numpy==1.22.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
