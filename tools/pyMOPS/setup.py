from setuptools import setup

setup(
    name="pyMOPS",
    version="0.1.0",
    author="Yongfeng Qiu",
    description="Python bindings for MOPS Ocean Visualization",
    packages=["pyMOPS"],
    package_dir={"pyMOPS": "pyMOPS"},
    package_data={"pyMOPS": ["*.so"]},
    zip_safe=False,
    python_requires=">=3.7",
)
