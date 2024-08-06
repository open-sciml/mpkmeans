import setuptools
import numpy
import platform
import importlib
import logging

PRJECT_NAME = "mpkmeans"
PACKAGE_NAME = "mpkmeans"
VERSION = "0.0.1"
SETREQUIRES=["numpy"]
MAINTAINER="Erin Carson, Xinye Chen, Xiaobo Liu"
EMAIL="xinyechenai@gmail.com"
INREUIRES=["numpy>=1.7.2", 'classixclustering', 'pychop', 'scikit-learn']


AUTHORS="InEXASCALE"

with open("README.md", 'r') as f:
    long_description = f.read()

ext_errors = (ModuleNotFoundError, IOError, SystemExit)
logging.basicConfig()
log = logging.getLogger(__file__)

if platform.python_implementation() == "PyPy":
    NUMPY_MIN_VERSION = "1.19.2"
else:
    NUMPY_MIN_VERSION = "1.17.2"
   
    
metadata = {"name":PRJECT_NAME,
            'packages':{"mpkmeans"},
            "version":VERSION,
            "setup_requires":SETREQUIRES,
            "install_requires":INREUIRES,
            "include_dirs":[numpy.get_include()],
            "long_description":long_description,
            "author":AUTHORS,
            "maintainer":MAINTAINER,
            "author_email":EMAIL,
            "classifiers":[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3"
            ],
            "maintainer_email":EMAIL,
            "description":"Python code for simulating low precision floating-point arithmetic",
            "long_description_content_type":'text/markdown',
            "url":"https://github.com/open-sciml/mpkmeans.git",
            "license":'MIT License'
}
            

class InvalidVersion(ValueError):
    """raise invalid version error"""

    
def check_package_status(package, min_version):
    """
    check whether given package.
    """
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = package_version >= min_version
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = "mpkmeans requires {} >= {}.\n".format(package, min_version)

    if package_status["up_to_date"] is False:
        if package_status["version"]:
            raise ImportError(
                "Your installation of {} {} is out-of-date.\n{}".format(
                    package, package_status["version"], req_str
                )
            )
        else:
            raise ImportError(
                "{} is not installed.\n{}{}".format(package, req_str)
            )


def setup_package():
    check_package_status("numpy", NUMPY_MIN_VERSION)
    
    setuptools.setup(
        **metadata
    )
    


if __name__ == "__main__":
    try:
        setup_package()
    except ext_errors as ext:
        log.warning(ext)
        log.warning("failure Installation.")
