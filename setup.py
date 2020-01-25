import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
	requirements = f.read()

setuptools.setup(
    name="shotTypeML_pkg", # Replace with your own username
    version="1.0.0",
    author="Thomas Koch",
    description="Shot type classification for historical videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/einthomas/CVSP2019-Project-01a-Shot-Type-Classification-dl-based",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
