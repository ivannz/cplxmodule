import os
from setuptools import setup


if __name__ == "__main__":
    # update the version number from the file at the root
    version = open("VERSION", "r").read().strip()

    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cwd, "cplxmodule", "__version__.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')

    setup(
        name="cplxmodule",
        version=version,
        description="Variational Dropout and Complex-valued Neural Networks in pytorch",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/ivannz/cplxmodule",
        keywords=[
            "pytorch",
            "variational dropout",
            "complex-valued networks",
            "sparsification",
        ],
        license="MIT License",
        author="Ivan Nazarov",
        author_email="ivan.nazarov@skolkovotech.ru",
        packages=[
            "cplxmodule",
            "cplxmodule.nn",
            "cplxmodule.nn.modules",
            "cplxmodule.nn.relevance",
            "cplxmodule.nn.relevance.real",
            "cplxmodule.nn.relevance.complex",
            "cplxmodule.nn.relevance.extensions",
            "cplxmodule.nn.relevance.extensions.real",
            "cplxmodule.nn.masked",
            "cplxmodule.nn.utils",
            "cplxmodule.utils",
        ],
        python_requires=">=3.7",
        install_requires=[
            "torch>=1.8",
            "numpy",
            "scipy",
        ],
        tests_require=[
            "tqdm",
            "torchvision",
            "matplotlib",
            "pytest",
            "scikit-learn",
        ],
    )
