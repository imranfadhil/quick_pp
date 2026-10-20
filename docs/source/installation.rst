.. highlight:: shell

============
Installation
============


Stable release
--------------

To install the core ``quick-pp`` petrophysical library (preferring `uv`_ or ``pip``), run:

.. code-block:: console

    $ uv pip install quick-pp
    # or modular install targets:
    $ uv pip install "quick-pp[core]"
    $ uv pip install "quick-pp[ml]"
    $ uv pip install "quick-pp[app]"
    $ uv pip install "quick-pp[all]"

Or with standard ``pip``:

.. code-block:: console

    $ pip install quick-pp[core]
    $ pip install quick-pp[ml]
    $ pip install quick-pp[app]
    $ pip install quick-pp[all]

.. _uv: https://github.com/astral-sh/uv


From sources
------------

The sources for ``quick_pp`` can be downloaded from the `Github repo`_.

Clone the public repository and install in editable mode using ``uv``:

.. code-block:: console

    $ git clone https://github.com/imranfadhil/quick_pp.git
    $ cd quick_pp
    $ uv venv --python 3.11
    $ uv pip install -e ".[all]"


.. _Github repo: https://github.com/imranfadhil/quick_pp
