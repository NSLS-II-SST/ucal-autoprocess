rom setuptools import setup, find_packages

setup(
    author="ctitus@bnl.gov",
    python_requires=">=3.7",
    description="Software to post-process raw TES data at SST1-ucal",
    install_requires=["tiled", "numpy", "databroker", "matplotlib", "ucalpost", "json", "mass"],
    packages=find_packages(),
    name="autoprocess",
    use_scm_version=True,
)
