#!/usr/bin/env python
# Steps:
# 1. "python -m build" to create new version of source archive and wheel in dist/
# 2. "python -m twine upload dist/*" to upload to PyPI or "twine upload -r testpypi dist/*" to upload to TestPyPI
# 2.a. Username: __token__
# 2.b. Password: <token>

"""The setup script."""

from setuptools import setup, find_packages
import quick_pp

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0', 'numpy==1.26.2', 'scikit-learn==1.6.1', 'matplotlib==3.8.0', 'SALib==1.5.1',
    'langchain_ollama==0.3.3', 'chainlit==2.5.5', 'ruptures', 'welly', 'pandas', 'statsmodels', 'mlflow',
    'fastapi', 'fastapi_mcp', 'psycopg2-binary', 'ptitprince', 'orjson'
]

test_requirements = []

setup(
    name='quick_pp',
    version=quick_pp.__version__,
    author=quick_pp.__author__,
    author_email=quick_pp.__email__,
    url='https://github.com/imranfadhil/quick_pp',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
    ],
    description="Python package to assist in providing quick-look/ preliminary petrophysical estimation.",
    entry_points={
        'console_scripts': [
            'quick_pp=main:cli',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='quick_pp, petrophysics, geoscience, well log analysis',
    packages=find_packages(include=['quick_pp', 'quick_pp.*']),  # where='quick_pp'),  #
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
)
