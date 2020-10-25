from setuptools import setup

setup(name='pytorch_indexing',
      version='0.1',
      description='Functions for efficient, large-scale, elementwise tensor-tensor comparisons with PyTorch Autograd support.',
      url='https://github.com/taylorpatti/pytorch_masks',
      author='Taylor Lee Patti',
      author_email='taylorpatti@g.harvard.edu',
      license='GNU LGPL',
      packages=['pytorch_masks'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
          'torch'
      ],
      zip_safe=False)
