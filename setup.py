from setuptools import setup, find_packages

setup(
    name="DifferentialGeometryTools",
    version="0.1.0",
    author="Daniel Lopez Cano",
    author_email="daniellopezcano13@gmail.com",
    description="A Python package for differential geometry utilities, manifolds, and visualization tools.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/daniellopezcano/DifferentialGeometryTools",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "jax",
        "matplotlib",
        "plotly",  # Optional but useful for interactive 3D plots
    ],
    python_requires='>=3.8',
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
