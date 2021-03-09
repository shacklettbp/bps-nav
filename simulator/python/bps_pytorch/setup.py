from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

curdir = os.path.dirname(os.path.realpath(__file__))
srcdir = os.path.dirname(os.path.dirname(curdir))

setup(
    name="bps_pytorch",
    ext_modules=[
        CUDAExtension(
            name="bps_pytorch",
            sources=[os.path.join(srcdir, "pytorch.cpp")],
            extra_compile_args=[],
            extra_link_args=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
