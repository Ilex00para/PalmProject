from setuptools import setup, find_packages

setup(
    name="palmproject_24",
    version="0.1",
    author="Ilex00para",
    description="A brief description",
    long_description=open('README.md').read(),  # You can include a detailed description
    long_description_content_type="text/markdown",
    url="https://github.com/Ilex00para/PalmProject.git",  # Project's GitHub or website URL
    packages=find_packages(),  # Automatically find and include all packages in your project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12.2',  # Minimum Python version required
)
