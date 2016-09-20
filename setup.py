import os

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
from glob import glob
import fnmatch

long_description = """.. -*-rst-*-

MDF - Data Flow Programming Toolkit
=======================================

"""

version = '2.2.1'
cython_profile = False
cdebug = False


# markers used by the preprocessor
_pure_python_start = "PURE PYTHON START"
_pure_python_end = "PURE PYTHON END"


requirements = []
with open("requirements.txt", "rb") as f:
    for line in f.readlines():
        if not line.strip() or line.startswith("#"):
            continue
        if "#egg=" in line:
            line = line.split("#egg=")[-1]
        requirements.append(line.strip())


def _preprocess(src_file):
    """
    Preprocess the python source files before cythoning.
    This is to work around limitations of using cython.compiled
    to conditionally declare code to only run when not compiled
    as the compiler complains it can't generate code to modify
    constants (eg conditional imports that have been cimported).
    """
    dest_file, ext = os.path.splitext(src_file)
    dest_file += ".cython" + ext

    if os.path.exists(dest_file) and os.stat(src_file).st_mtime < os.stat(dest_file).st_mtime:
        return dest_file

    output_lines = []
    with open(src_file) as src_fh:
        skip = False
        for line in src_fh.readlines():
            if line.startswith("#") and line.lstrip("# ").upper().startswith(_pure_python_start):
                skip = True

            if not skip:
                output_lines.append(line)

            if line.startswith("#") and line.lstrip("# ").upper().startswith(_pure_python_end):
                skip = False

    with open(dest_file, "wt") as dest_fh:
        dest_fh.writelines(output_lines)

    return dest_file


if __name__ == "__main__":

    extra_compile_args = []
    extra_link_args = []
    if cdebug:
        extra_compile_args = ["/Zi"]
        extra_link_args = ["/DEBUG"]

    ext_modules = []
    for dirpath, dirnames, files in os.walk("mdf"):
        for file in fnmatch.filter(files, "*.pxd"):
            basename = os.path.join(dirpath, os.path.splitext(file)[0])
            module = ".".join(basename.split(os.path.sep))
            src = basename + ".py"
            src = _preprocess(src)
            ext_modules.append(Extension(module, [src]))

    for e in ext_modules:
        e.pyrex_directives = {"profile": cython_profile}
        e.extra_compile_args.extend(extra_compile_args)
        e.extra_link_args.extend(extra_link_args)

    setup(
        name='mdf',
        version=version,
        description='MDF - Data Flow Programming Toolkit',
        long_description=long_description,
        zip_safe=False,

        # The icons directory is not a python package so find_packages will not find it.
        packages=find_packages() + ["mdf.viewer.icons"],
        package_data={'mdf.viewer.icons': ['*.ico']},
        test_suite='nose.collector',
        setup_requires=[],
        scripts=glob("bin/*.py"),
        install_requires=requirements,
        extras_require={
            'win32': ['pywin32'],
            'linux2': []
        },
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_modules,
    )
