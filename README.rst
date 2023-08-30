
==================================================
FLORAH - generate assembly history with flow-based recurrent model
==================================================

FLORAH generate assembly history of halos using a recurrent neural network and normalizing flow model, Nguyen et al. (2023) [1]_. You can also find our paper on arXiv at `https://arxiv.org/abs/2308.05145`.

:Authors:
    Tri Nguyen,
    Chirag Modi,
    L.Y. Aaron Yung,
    Rachel S. Somerville
:Maintainer:
    Tri Nguyen (tnguy@mit.edu)
:Version: 0.0.0 (2023-08-30)

Installation
------------

To install FLORAH, simply clone the repo and install with `pip`:

.. code-block:: bash

    git clone https://github.com/trivnguyen/florah.git
    pip install .

This should install all the dependencies as well. If you want to install the dependencies separately, please see the section below.

Dependencies
------------

The following dependencies are required to run this project:

- Python 3.6 or later
- NumPy 1.22.3 or later
- SciPy 1.9.1 or later
- Astropy 5.2.2 or later
- PyTorch Lightning 1.7.6 or later
- PyTorch 2.0.0 or later

To install the dependencies separately, you can use `pip`:

.. code-block:: bash

    pip install -r requirements.txt

It is recommended to use a virtual environment to manage the dependencies and avoid version conflicts. You can create a virtual environment and activate it using the following commands:

.. code-block:: bash

    python -m venv env
    source env/bin/activate (Linux/MacOS)
    env\Scripts\activate.bat (Windows)

Once the virtual environment is activated, you can install the dependencies using pip as usual.

Usage
-----
An example training and generation Jupyter Notebook can be found at ``tutorials/example_training.ipynb``.

The rest of the tutorials are under construction. More to come!

Documentation
-------------

Under construction.

Contributing
------------

We welcome contributions to FLORAH! To contribute, please contact Tri Nguyen (tnguy@mit.edu).

License
-------

FLORAH is licensed under the MIT license. See ``LICENSE.md`` for more information.

References
----------
.. [1] Tri Nguyen, Chirag Modi, L.Y. Aaron Yung, Rachel S. Somerville, "FLORAH: A generative model for halo assembly histories", arXiv e-prints, 2023, https://doi:10.48550/arXiv.2308.05145
