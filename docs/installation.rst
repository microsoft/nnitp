Installation
============

Nnitp is known to work with Python versions 3.6 and 3.7. It may or
may not work with Python 3.8, depending on what versions of various
packages are available for your platform.

Ubuntu and other Debian-based Linux
-----------------------------------

Install Git and Python 3::

    $ apt install git python3

Download the source files from Github::

    $ git clone https://github.com/microsoft/nnitp

Install the `nnitp` Python package::

    $ cd nnitp
    $ pip3 install .

Windows 10
----------

Install `Git bash <https://gitforwindows.org/>`_ and `Python 3.7
<https://www.python.org/downloads/windows/>`_. When you run the Python
installer, make sure to check the box to install Python directories in
your PATH. Earlier versions of Python, might work, but 3.8 does not
because the necessary version of Tensorflow is not available.

In a Git bash window:

    $ cd /c
    $ git clone https://github.com/microsoft/nnitp

In a Windows cmd window:

    > cd c:\
    > pip install .


  

