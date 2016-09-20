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


requirements = []
with open("requirements.txt", "rb") as f:
    for line in f.readlines():
        if not line.strip() or line.startswith("#"):
            continue
        if "#egg=" in line:
            line = line.split("#egg=")[-1]
        requirements.append(line.strip())


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
            ext_modules.append(Extension(module, [basename + ".py"]))

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
