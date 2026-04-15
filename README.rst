.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/pyamapping.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/pyamapping
    .. image:: https://readthedocs.org/projects/pyamapping/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://pyamapping.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/conda/vn/conda-forge/pyamapping.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/pyamapping
    .. image:: https://pepy.tech/badge/pyamapping/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/pyamapping
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/pyamapping

.. image:: https://img.shields.io/pypi/v/pyamapping.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/pyamapping/

.. image:: https://img.shields.io/coveralls/github/interactive-sonification/pyamapping/main.svg
    :alt: Coveralls
    :target: https://coveralls.io/r/interactive-sonification/pyamapping

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==========
pyamapping
==========


    A Python package containing audio related functions and mappings



.. _pyscaffold-notes:

Making Changes & Contributing
=============================

To setup your development environment, you need to have `uv <https://docs.astral.sh/uv/>`_ installed.
First, clone the repository and then change into the root directory:

.. code-block:: bash

    cd pyamapping

Afterwards, sync your environment:

.. code-block:: bash

    uv sync

This will install all required dependencies (dev dependencies included).
Also, please install the pre-commit hooks:

.. code-block:: bash

    uv run pre-commit install

Experimentation in Scratchpad / Jupyter
----------------------------------------

uv automatically installs the ``pyamapping`` package to the ``.venv`` in editable mode.
During development, any changes to the ``pyamapping`` source will be reflected automatically.
To write playground / exploratory code, just make sure to run your REPL environment with ``uv run``, e.g.:

.. code-block:: bash

    uv run python

or

.. code-block:: bash

    uv run jupyter notebook

Then, you should be able to:

.. code-block:: python

    import pyamapping as pam

And get the current state of the source code.

Running Tests
-------------

Tests can be run via:

.. code-block:: bash

    uv run tox -e tests

Building Documentation
----------------------

Documentation can be built via:

.. code-block:: bash

    uv run tox -e docs

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
