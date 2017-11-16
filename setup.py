import  os
import sys
from os.path import join, dirname, realpath, relpath, splitext, abspath, exists, getmtime, relpath, lexists, islink
from os import walk, sep, remove, listdir, stat, symlink, pathsep
from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
import warnings
import tempfile
import subprocess

import numpy as np
import tensorflow as tf

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    solutions = []
    for dir in path.split(pathsep):
        binpath = join(dir, name)
        if exists(binpath):
            solutions.append(abspath(binpath))
    if len(solutions) >= 1:
        if any("usr" in sol for sol in solutions):
            solutions = [sol for sol in solutions if "usr" in sol]
        return solutions[0]
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = join(home, "bin", "nvcc")
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        print()
        if nvcc is None:
            return None
        home = dirname(dirname(nvcc))
        print(home)
    cudaconfig = {"nvcc": nvcc,
                  "include": [join(home, "include"), join(home, "include", "cuda")],
                  "lib64": [join(home, "lib64"), join(home, "lib")]}
    for k, v in cudaconfig.items():
        if isinstance(v, str):
            v = [v]
        all_missing = all(not exists(path) for path in v)
        if all_missing:
            raise EnvironmentError("The CUDA %s path could not be located in %r" % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


def check_openmp_presence():
    source = """
        #include <omp.h>
        int main() {
        #ifdef _OPENMP
          return 0;
        #else
          breaks_on_purpose
        #endif
        }
    """
    with tempfile.NamedTemporaryFile() as foutput:
        with tempfile.NamedTemporaryFile() as ftest:
            with open(ftest.name, "wt") as fout:
                fout.write(source)
            try:
                out = subprocess.check_output(["g++", ftest.name, "-o", foutput.name, "-fopenmp"])
                return True
            except subprocess.CalledProcessError:
                return False



# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


def find_files_by_suffix(path, suffix):
    """Recursively find files with specific suffix in a directory"""
    for relative_path, dirs, files in walk(path):
        for fname in files:
            if fname.endswith(suffix):
                yield join(path, relative_path, fname)

CUDA = locate_cuda()
SRC_DIR = join(dirname(realpath(__file__)), "src")
TF_LIB = tf.sysconfig.get_lib()
TF_INCLUDE = tf.sysconfig.get_include()
TF_CUDA = tf.test.is_built_with_cuda()
HAS_OPENMP = check_openmp_presence()

if TF_CUDA and CUDA is None:
    warnings.warn("qrnn can run on gpu, but nvcc was not found in your path. "
                  "Either add it to your path, or set the $CUDAHOME variable.")

USE_CUDA = TF_CUDA and CUDA is not None


cu_sources = list(find_files_by_suffix(SRC_DIR, ".cu"))
cpp_sources = list(find_files_by_suffix(SRC_DIR, ".cpp"))


cmdclass = {}
include_dirs = [np.get_include(), TF_INCLUDE, SRC_DIR, join(SRC_DIR, "third_party")]
TF_FLAGS = ["-D_MWAITXINTRIN_H_INCLUDED", "-D_FORCE_INLINES", "-D_GLIBCXX_USE_CXX11_ABI=0"]
gcc_extra_compile_args = ["-g", "-std=c++11", "-fPIC", "-O3", "-march=native", "-mtune=native"] + TF_FLAGS

nvcc_extra_compile_args = []
extra_link_args = ["-fPIC"]
if HAS_OPENMP:
    gcc_extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")

if sys.platform == 'darwin':
    gcc_extra_compile_args.append('-stdlib=libc++')
    nvcc_extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')
else:
    extra_link_args.append("-shared")


if USE_CUDA:
    cmdclass["build_ext"] = custom_build_ext
    gcc_extra_compile_args.extend(["-D", "GOOGLE_CUDA"])
    include_dirs.extend(CUDA["include"])
    nvcc_extra_compile_args.extend(TF_FLAGS + ["-std=c++11", "-D", "GOOGLE_CUDA=1",
                               "-I", TF_INCLUDE,
                               "-x", "cu", "--compiler-options", "'-fPIC'",
                               "--gpu-architecture=sm_30", "-lineinfo",
                               "-Xcompiler", "-std=c++98"] + ["-I" + path for path in CUDA["include"]])
    extra_compile_args = {"gcc": gcc_extra_compile_args,
                          "nvcc": nvcc_extra_compile_args}
    runtime_library_dirs = CUDA['lib64']
else:
    cu_sources = []
    extra_compile_args = gcc_extra_compile_args
    runtime_library_dirs = []


ext = Extension("qrnn_lib",
                sources=cu_sources + cpp_sources,
                library_dirs=[TF_LIB],
                libraries=["tensorflow_framework"],
                language="c++",
                runtime_library_dirs=runtime_library_dirs,
                # this syntax is specific to this build system
                # we're only going to use certain compiler args with nvcc and not with gcc
                # the implementation of this trick is in customize_compiler() below
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                include_dirs=include_dirs)

setup(name='qrnn',
      # random metadata. there's more you can supploy
      author="Jonathan Raiman",
      author_email="jonathanraiman@gmail.com",
      version="0.2.2",
      install_requires=["numpy", "tensorflow>=1.4"],
      ext_modules = [ext],
      py_modules=["qrnn"],
      # inject our custom trigger
      cmdclass=cmdclass,
      zip=False)
