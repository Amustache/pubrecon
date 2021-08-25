import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(name='pubrecon',
                 version='0.1',
                 author="Hugo 'Stache' Hueber",
                 author_email="hugo.hueber@epfl.ch",
                 license='WTFPL',
                 description='Ad recognition',
                 long_description=readme(),
                 long_description_content_type="text/markdown",
                 url='http://github.com/AMustache/ML-2019/project2/',
                 packages=setuptools.find_packages(),
                 install_requires=['matplotlib>=3.1', 'tensorflow==2.5.1', 'opencv-python>=4.1',
                                   'opencv-contrib-python>=4.1', 'keras==2.2.4', 'numpy>=1.17', 'tqdm>=4.40',
                                   'pandas==0.25.3', 'sklearn'],
                 include_package_data=True,
                 zip_safe=False,
                 python_requires='==3.7.5',
                 )
