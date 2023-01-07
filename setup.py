from setuptools import setup, find_packages

with open("docs/description.md", "r", encoding="utf-8") as file:
    long_description = file.read() 

install_requires = [
    "joblib==1.2.0",
    "numpy==1.24.1",
    "pandas==1.5.2",
    "python-dateutil==2.8.2",
    "pytz==2022.7",
    "scikit-learn==1.2.0",
    "scipy==1.9.3",
    "six==1.16.0",
    "sklearn==0.0",
    "threadpoolctl==3.1.0",
    "tqdm==4.64.1",
    "matplotlib==3.6.2"
]

setup(
    name="cluster_ss",
    version="0.0.2",
    url="https://github.com/xGabrielR/cluster_ss",
    author="Gabriel R.",
    author_email="gabrielrichter2021@gmail.com",
    description="Improving Clustering Problem Analysis with a simple Silhouette Metric Support and Sklearn Estimators Fit's.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.10",
    keywords=["python", "first_package"]
)