from setuptools import find_packages, setup

setup(
    name="kunserve",
    version="0.0.1",
    author="IPADS",
    description="Inference engine for LLMs.",
    packages=find_packages(include=["kunserve", "kunserve.*"]),
    zip_safe=False,
)
