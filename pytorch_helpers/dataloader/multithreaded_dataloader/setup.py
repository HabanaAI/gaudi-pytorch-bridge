import setuptools

setuptools.setup(
    name="habana_torch_dataloader",
    version="0.0.0",
    #author="habana.ai",
    #author_email="@habana.ai",
    #license='',
    description="Package to create custom multithreaded dataloader for PyTorch",
    install_requires=[],
    python_requires='>=3.6',
    packages=setuptools.find_packages(
                                     exclude=('build',
                                              'dist',
                                              'habana_torch_dataloader.egg-info',)),
    classifiers=[ #'License :: Approved ::  License',
                 'Programming Language :: Python :: 3',
    ]
)
