:github_url: https://github.com/google/lightweight_mmm/tree/main/docs

LightweightMMM Documentation
===================

LightweightMMM 🦇 is a lightweight Bayesian Marketing Mix Modeling (MMM) 
library that allows users to easily train MMMs and obtain channel attribution 
information. The library also includes capabilities for optimizing media 
allocation as well as plotting common graphs in the field.

It is built in python3 and makes use of Numpyro and JAX.

Installation
------------

We have kept JAX as part of the dependencies to install when installing
LightweightMMM, however if you wish to install a different version of JAX or 
jaxlib for specific CUDA/CuDNN versions see 
https://github.com/google/jax#pip-installation for instructions on installing 
JAX. Otherwise our installation assumes a CPU setup.

The recommended way of installing lightweight_mmm is through PyPi:

``pip install --upgrade pip``
``pip install lightweight_mmm``

If you want to use the most recent and slightly less stable version you can install it from github:

``pip install --upgrade git+https://github.com/google/lightweight_mmm.git``


.. toctree::
   :caption: Model Documentation
   :maxdepth: 1

   models

.. toctree::
   :caption: Custom priors
   :maxdepth: 1

   custom_priors

.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api

.. toctree::
   :caption: FAQ
   :maxdepth: 2

   faq

Contribute
----------

- Issue tracker: https://github.com/google/lightweight_mmm/issues
- Source code: https://github.com/google/lightweight_mmm/tree/main

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/google/lightweight_mmm/issues>`_.

License
-------

LightweightMMM is licensed under the Apache 2.0 License.

Indices and tables
==================

* :ref:`genindex`