from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import multiprocessing
import subprocess
import os

class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class CustomBuilder(build_ext):
    def run(self):
        for ext in self.extensions:
            self.cmake(ext)

    def cmake(self, ext):
        srcdir = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))))
        outdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))

        build_type = "Debug" if self.debug else "RelWithDebInfo"
        cmake_args = [
            f'-S{srcdir}',
            f'-B{self.build_temp}',
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={outdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={outdir}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        ]

        if not os.path.exists(
                os.path.join(self.build_temp, "CMakeCache.txt")):
            subprocess.check_call(["cmake"] + cmake_args)

        subprocess.check_call(["cmake", "--build", f'{self.build_temp}',
                               "--parallel", str(multiprocessing.cpu_count())])

setup(
    name="bps_sim",
    ext_modules=[
        CMakeExtension(
            name="bps_sim"
        )
    ],
    cmdclass={"build_ext": CustomBuilder}
)
