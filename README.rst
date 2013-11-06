This is pysnopt, a Python interface to SNOPT.

Copyright (C) 2013  Manuel Kudruss, Sebastian F. Walter


Contact the authors
-------------------

* sebastian.walter@gmail.com
* manuel.kudruss@gmail.com


LICENSE
-------

GPL v3

See LICENSE.txt for details.


INSTALLATION
------------

known to work with:

    * gcc version 4.6.3
    * SNOPT 7
    * Cython version 0.19
    * numpy version 1.6.2


Copy the contents of this folder to your SNOPT root directory::

    SNOPT7/
          - ...
          - cppsrc/
          - csrc/
          - python/
                - README.txt
                - Makefile
                - setup.py
                - snopt.pxd
                - snopt.pyx
                - ...
          - ...

Then::

    cd SNOPT7/python
    make
    make python
    python test.py
    python toy_example.py

If you have any issues, please file a bug report or send us an email.