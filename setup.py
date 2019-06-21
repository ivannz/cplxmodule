from distutils.core import setup, Extension

setup(
    name="cplxmodule",
    version="0.6.8",
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
