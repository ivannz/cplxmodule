from distutils.core import setup, Extension

setup(
    name="cplxmodule",
    version="0.9.7",
    description="""A lightweight extension for pytorch.nn for handling """
                """complex valued computations.""",
    license="MIT License",
    author="Ivan Nazarov",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "cplxmodule",
        "cplxmodule.nn",
        "cplxmodule.nn.relevance",
        "cplxmodule.nn.masked",
        "cplxmodule.nn.utils",
        "cplxmodule.utils",
    ],
    install_requires=[
        "torch>=1.4",
        "numpy",
        "scipy"
    ],
    tests_require=[
        "tqdm",
        "torchvision",
        "matplotlib",
        "pytest"
    ]
)
