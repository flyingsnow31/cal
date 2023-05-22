from setuptools import setup, Extension

module1 = Extension('cal',
                    sources=['decision_tree/libDecisionTree.cpp', 'pyWarpper.cpp',
                             'simple_net/libNet.cpp', 'simple_net/Function.cpp', ],
                    include_dirs=['/home/lzy/anaconda3/lib/python3.9/site-packages/pybind11/include',
                                  '/usr/local/include/opencv4/', '/usr/local/include/'],
                    libraries=['opencv_stitching', 'opencv_core', 'opencv_objdetect', 'opencv_calib3d',
                               'opencv_features2d', 'opencv_highgui', 'opencv_videoio', 'opencv_imgcodecs',
                               'opencv_video', 'opencv_photo', 'opencv_ml', 'opencv_imgproc', 'opencv_flann'
                               ],
                    library_dirs=['/usr/local/lib'],
                    extra_compile_args=['-std=c++2a'])

setup(name='cal',
      version='1.0',
      description='This is a cal, dt and net',
      ext_modules=[module1],
      author="cq and lzy",
      python_requires=">=3.6")
