from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='pubrecon',
      version='0.1',
      description='Ad recognition',
      long_description=readme(),
      url='http://github.com/AMustache/ML-2019/project2/',
      author='Stache',
      author_email='stache@hi2.in',
      packages=['pubrecon'],
      install_requires=[],
      include_package_data=True,
      zip_safe=False)
