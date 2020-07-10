from setuptools import setup

setup(
    name="cplxmodule",
    version="2020.08",
    description="Variational Dropout and Complex-valued Neural Networks in pytorch",
    long_description="An extension for pytorch providing essential building blocks "
                     "for Complex-valued Neural Networks and `torch.nn`-compatible "
                     "real- and complex-valued Variational Dropout methods. "
                     "Implements the tools necessary to train and sparsify "
                     "both real and complex-valued models, and seamlessly "
                     "integrate them into existing pipelines.",
    url="https://github.com/ivannz/cplxmodule",
    keywords=[
        "pytorch",
        "variational dropout",
        "complex-valued networks",
        "sparsification"
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
