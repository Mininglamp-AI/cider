"""Build script for Cider.

Usage:
    pip install -e .          # Editable install (dev)
    pip install .             # Regular install
    python setup.py build_ext # Build C++ extension only

The C++ extension is built via CMake and installed into cider/lib/.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        src_dir = Path(__file__).parent.resolve()
        build_dir = src_dir / "build"
        lib_dir = src_dir / "cider" / "lib"
        build_dir.mkdir(exist_ok=True)
        lib_dir.mkdir(exist_ok=True)

        cmake_args = [
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build_args = ["--config", "Release", "-j"]

        subprocess.check_call(
            ["cmake", str(src_dir)] + cmake_args,
            cwd=build_dir,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir,
        )

        # Copy artifacts to cider/lib/
        import shutil
        for f in build_dir.glob("*.so"):
            shutil.copy2(f, lib_dir)
        for f in build_dir.glob("*.dylib"):
            shutil.copy2(f, lib_dir)

        # Also copy .so to where setuptools expects it (for pip install -e .)
        ext_dest = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_dest), exist_ok=True)
        so_files = list(build_dir.glob("_cider_prim*.so"))
        if so_files:
            shutil.copy2(so_files[0], ext_dest)


setup(
    ext_modules=[Extension("_cider_prim", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
)
