from setuptools import setup, find_packages
setup(
    name="fragdockrl",
    version="0.1.0",
    author="Seung Hwan Hong",
    author_email="shhong@novelismlab.com",
    description="Fragment-based docking with reinforcement learning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/novelism/FragDockRL",
    packages=find_packages(exclude=["examples*", "scripts*", "data*"]),
    include_package_data=True,
    classifiers=[
                "Programming Language :: Python :: 3",
                "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy==1.26",
        "pandas==2.3.1",
        "rdkit==2023.09.6",
        "smina==2020.12.10",
        "torch==2.7.1",
        "AutoDockTools_py3 @ git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git"
    ],
    python_requires=">=3.12",
)

