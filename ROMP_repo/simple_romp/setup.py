import setuptools

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

try:
    import numpy
except ImportError:
    numpy = None

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requireds = [
    "opencv-python-headless",
    "torch",
    'setuptools>=18.0.0',
    'cython',
    'numpy>=1.21.0',
    'typing-extensions>=4.1',
    'scipy',
    'lap'
]

setuptools.setup(
    name='simple_romp',
    version='1.0.6',
    author="Yu Sun",
    author_email="yusun@stu.hit.edu.cn",
    setup_requires=[
        'setuptools>=18.0.0',
        'cython',
        'numpy>=1.21.0',
        'typing-extensions>=4.1',
        'scipy',
        'lap'
    ],
    install_requires=requireds,
    description="ROMP: Monocular, One-stage, Regression of Multiple 3D People, ICCV21",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arthur151/ROMP",
    packages=[
        'romp',
        'vis_human',
        'bev',
        'tracker',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/Arthur151/ROMP/issues",
    },
    entry_points={
        "console_scripts": [
            "romp=romp.main:main",
            "bev=bev.main:main",
        ],
    },
)
