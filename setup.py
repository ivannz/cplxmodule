from distutils.core import setup, Extension

setup(
    name="cplxmodule",
    version="2020.03",
    description="An extension for pytorch providing essential building blocks"
                " for complex-valued networks and `torch.nn`-compatible"
                " Bayesian sparsification methods. Implements the tools"
                " necessary to train and sparsify both real and complex-valued"
                " models, and seamlessly integrate them into existing models.",
    license="MIT License",
    author="Ivan Nazarov",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "cplxmodule",
        "cplxmodule.nn",
        "cplxmodule.nn.modules",
        "cplxmodule.nn.relevance",
        "cplxmodule.nn.relevance.extensions",
        "cplxmodule.nn.relevance.extensions.real",
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
