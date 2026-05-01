"""Build script for Cider.

Usage:
    pip install -e .          # Editable install (dev)
    pip install .             # Regular install
    python setup.py build_ext # Build C++ extension only

The C++ extension is built via CMake and installed into cider/lib/.
On Apple M4 and below, the C++ extension is skipped (INT8 TensorOps
require M5+). The pure-Python package still installs and provides
graceful fallback via is_available() → False.
"""

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def _detect_apple_chip() -> int:
    """Detect Apple Silicon generation. Returns chip number (4, 5, ...) or 0 if unknown."""
    if platform.system() != "Darwin":
        return 0
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        m = re.match(r"Apple M(\d+)", brand)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def _should_build_ext() -> bool:
    """Determine if C++ extension should be built."""
    # Allow forcing via env var
    force = os.environ.get("CIDER_FORCE_BUILD", "").lower()
    if force in ("1", "true", "yes"):
        return True
    if force in ("0", "false", "no"):
        return False
    # Auto-detect: only build on M5+
    chip = _detect_apple_chip()
    if chip >= 5:
        return True
    if chip > 0:
        print(f"[cider] Detected Apple M{chip} — skipping C++ extension (requires M5+)")
        return False
    # Unknown chip (cross-compile, CI, etc.) — try to build
    return True


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


if _should_build_ext():
    ext_modules = [Extension("_cider_prim", sources=[])]
    cmdclass = {"build_ext": CMakeBuild}
else:
    ext_modules = []
    cmdclass = {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
