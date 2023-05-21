from setuptools import setup, Extension

module1 = Extension('cal',
                    sources=['decision_tree/libDecisionTree.cpp', 'pyWarpper.cpp'],
                    include_dirs=['/home/lzy/anaconda3/lib/python3.9/site-packages/pybind11/include'],
                    extra_compile_args=['-std=c++2a'])

setup(name='cal',
      version='1.0',
      description='This is a cal, dt and net',
      ext_modules=[module1],
      author="cq and lzy",
      python_requires=">=3.6")
