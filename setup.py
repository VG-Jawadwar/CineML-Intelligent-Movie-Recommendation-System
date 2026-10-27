from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# REPO_NAME = "CineML - Intelligent Movies Recommender System"
AUTHOR_NAME = "Vaibhav Jawadwar"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ['streamlit']


setup(
    name=SRC_REPO,
    version="0.0.1",
    author = AUTHOR_NAME,
    author_email="vaibhavjawadwar5@gmail.com",
    description="A small local packages for ML based movies recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[SRC_REPO],
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)