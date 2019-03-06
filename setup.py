from setuptools import setup

setup(name='backprop',
      version='0.1',
      description='building neural nets from scratch',
      author='Brian Farris',
      packages=['backprop'],
      install_requires=["numpy", "matplotlib", "sklearn"],
      zip_safe=False)
