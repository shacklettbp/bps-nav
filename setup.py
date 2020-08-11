from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="v4r_pytorch",
    ext_modules=[
        CUDAExtension(
            name="v4r_pytorch",
            sources=["pytorch.cpp"],
            extra_compile_args=[],
            extra_link_args=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
