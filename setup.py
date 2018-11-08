import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorsketch",
    version="0.0.1",
    author="Yang Guo, Yiming Sun, Charlene Luo",
    author_email="yg93@cornell.edu, ys784@cornell.edu, cl894@cornell.edu",
    description="Implementation of two-pass and one-pass tensor sketching algorithms",
    long_description="Implementation of two-pass and one-pass tensor sketching algorithms",
    long_description_content_type="text/markdown",
    url="https://github.com/sunstat/SketchTensor",
    test_suite="nose.collector",
    tests_require=['nose'], 
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy', 
        'tensorly', 
        'netCDF4', 
        'matplotlib',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)


