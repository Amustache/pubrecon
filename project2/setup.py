import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(name='pubrecon',
                 version='0.1',
                 author="Hugo 'Stache' Hueber",
                 author_email="hugo.hueber@epfl.ch",
                 license='MIT',
                 description='Ad recognition',
                 long_description=readme(),
                 long_description_content_type="text/markdown",
                 url='http://github.com/AMustache/ML-2019/project2/',
                 packages=setuptools.find_packages(),
                 install_requires=['mathplotlib', 'tensorflow', 'opencv-python', 'opencv-contrib-python', 'keras', 'numpy', 'tqdm', 'pandas', 'sklearn'],
                 include_package_data=True,
                 zip_safe=False,
                 python_requires='>=3.7',
                 )
