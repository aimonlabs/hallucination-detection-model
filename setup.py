from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aimon-hdm2",
    version="0.1.0",
    author="Bibek Paudel",
    author_email="bibek@aimon.ai",
    description="HDM2: Hallucination Detection Model by AimonLabs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aimonlabs/hallucination-detection-model",
    project_urls={
        "Bug Tracker": "https://github.com/aimonlabs/hallucination-detection-model/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["accelerate"],
    },
)