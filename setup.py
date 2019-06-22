from distutils.core import setup, Extension

setup(
    name="cplxmodule",
    version="0.7.0",
    description="""A lightweight extension for pytorch.nn for handling """
                """complex valued computations.""",
    license="MIT License",
    author="Ivan Nazarov",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "cplxmodule",
        "cplxmodule.utils",
        "cplxmodule.relevance",
        "cplxmodule.masked"
    ],
    requires=["torch", "numpy"]
)
