import pathlib
from setuptools import setup, find_packages


# Read requirements from requirement.txt
with pathlib.Path('requirement.txt').open() as requirements_txt:
    install_requires = [
        line.strip()
        for line in requirements_txt
        if line.strip() and not line.startswith('#')
    ]


setup(
    name='colm',
    packages=find_packages(),
    version='0.1',
    description='CoLM - Sequential Multi-Task Learning',
    author='Dang Nguyen',
    url='https://github.com/hsgser/CoLM',
    install_requires=install_requires,
    entry_points={
        "console_scripts": [],
    },
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)
