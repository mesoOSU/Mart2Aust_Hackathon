======================
|Auslogo| Mart2Aust
======================

.. |Auslogo| image:: https://raw.githubusercontent.com/argerlt/Mart2Aust_Hackathon/master/doc/_static/img/AusRecon_logo.png
   :width: 100 

Mart2Aust is a port of the Matlab Proect AusRecon written in python. AusRecon
was originally written using MTEX, and as Orix is the closest comaparable
tool in python, AusRecon is built with Orix as the base. If this project in 
successful or useful, this fork will eventually become a pull request for 
Orix.

In the meantime, any work based off of this repo should cite the manuscript 
at the following `link <https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576720011103>`_,
as well as the `AusRecon Paper <https://link.springer.com/article/10.1007/s11661-019-05514-4>`_.

The current Alpha version of AusRecon can be found `Here <https://github.com/mesoOSU/AusRecon>`_

Below this point is the original orix Readme.rst 
---

===========
|logo| orix
===========

.. |logo| image:: https://raw.githubusercontent.com/pyxem/orix/master/doc/_static/img/orix_logo.png
   :width: 100

|binder|_ |build_status|_ |Coveralls|_ |docs|_ |pypi_version|_  |downloads|_ |black|_ |doi|_

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/pyxem/orix/HEAD

.. |build_status| image:: https://github.com/pyxem/orix/workflows/build/badge.svg
.. _build_status: https://github.com/pyxem/orix/actions

.. |Coveralls| image:: https://coveralls.io/repos/github/pyxem/orix/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/pyxem/orix?branch=master

.. |docs| image:: https://readthedocs.org/projects/orix/badge/?version=latest
.. _docs: https://orix.readthedocs.io/en/latest

.. |pypi_version| image:: http://img.shields.io/pypi/v/orix.svg?style=flat
.. _pypi_version: https://pypi.python.org/pypi/orix

.. |downloads| image:: https://anaconda.org/conda-forge/orix/badges/downloads.svg
.. _downloads: https://anaconda.org/conda-forge/orix

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _black: https://github.com/psf/black

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3459662.svg
.. _doi: https://doi.org/10.5281/zenodo.3459662

orix is an open-source python library for analysing orientations and crystal symmetry.

The package defines objects and functions for the analysis of orientations represented
as quaternions or 3D rotation vectors accounting for crystal symmetry. Functionality
builds primarily on top of `numpy <http://www.numpy.org/>`_ and
`matplotlib <https://matplotlib.org/>`_ and is heavily inspired by the
MATLAB package `MTEX <http://mtex-toolbox.github.io/>`_.

If analysis using orix forms a part of published work please cite the manuscript
at the following
`link <https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576720011103>`_.
You can also find demos in the
`orix-demos <https://github.com/pyxem/orix-demos>`_ repo, visit the `user guide
<https://orix.readthedocs.io>`_, or try out the user guide notebooks
interactively in the browser by clicking the Binder link above and navigating
to the ``doc/`` directory.

orix is released under the GPL v3 license.
