from setuptools import find_packages, setup
import os

# # change / add these as needed
# if LooseVersion(sys.version) < LooseVersion("3.x"):
#     raise RuntimeError(
#         "{{ cookiecutter.project_name  }} requires python >= 3.x, "
#         "but your Python version is {}".format(sys.version)
#     )

# if LooseVersion(pip.__version__) < LooseVersion("xx"):
#     raise RuntimeError(
#         "pip>=xx.x.x is required, but your pip version is {}. "
#         'Try again after "pip install -U pip"'.format(pip.__version__)
#     )

# requirements = {
#     "install": ["tensorflow-gpu>=2.x", "tensorflow-addons>=0.x"],
#     "setup": [],
#     "test": [],
# }

# install_requires = requirements["install"]
# setup_requires = requirements["setup"]
# tests_require = requirements["test"]

{%- set license_classifiers = {
    'MIT license': 'License :: OSI Approved :: MIT License',
    'BSD license': 'License :: OSI Approved :: BSD License',
    'ISC license': 'License :: OSI Approved :: ISC License (ISCL)',
    'Apache Software License 2.0': 'License :: OSI Approved :: Apache Software License',
    'GNU General Public License v3': 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
} %}

setup(
    name='{{ cookiecutter.repo_name  }}',
    packages=find_packages(include=["{{ cookiecutter.source }}*"]),
    version='{{ cookiecutter.version }}',
    description='{{ cookiecutter.description }}',
    long_description=open(os.path.join('{{ cookiecutter.repo_name  }}', "README.rst"), encoding="utf-8").read(),
    long_description_content_type="text/x-rst",
    author='{{ cookiecutter.full_name }}',
    author_email='{{ cookiecutter.email }}',
{%- if cookiecutter.open_source_license in license_classifiers %}
    license="{{ cookiecutter.open_source_license }}",
{%- endif %}
    url='https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.repo_name }}',
    # install_requires=install_requires,
    # setup_requires=setup_requires,
    # tests_require=tests_require,
)

