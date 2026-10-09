.. _installation:

Installation
============

.. contents::
    :local:
    :depth: 1

Install from Conda
------------------

.. warning::

   TODO: Prepare Conda package.

Install from GitHub
-------------------

Check out code from the Albatross GitHub repo and start the installation:

.. code-block:: console

   $ git clone https://github.com/climateintelligence/albatross.git
   $ cd albatross

Create Conda environment named `albatross`:

.. code-block:: console

   $ conda env create -f environment.yml
   $ conda activate albatross

Install Albatross app:

.. code-block:: console

  $ pip install -e .
  OR
  make install

For development you can use this command:

.. code-block:: console

  $ pip install -e .[dev]
  OR
  $ make develop

Start Albatross PyWPS service
-----------------------------

After successful installation you can start the service using the ``albatross`` command-line.

.. code-block:: console

   $ albatross --help # show help
   $ albatross start  # start service with default configuration

   OR

   $ albatross start --daemon # start service as daemon
   loading configuration
   forked process id: 42

The deployed WPS service is by default available on:

http://localhost:5000/wps?service=WPS&version=1.0.0&request=GetCapabilities.

.. NOTE:: Remember the process ID (PID) so you can stop the service with ``kill PID``.

You can find which process uses a given port using the following command (here for port 5000):

.. code-block:: console

   $ netstat -nlp | grep :5000


Check the log files for errors:

.. code-block:: console

   $ tail -f  pywps.log

... or do it the lazy way
+++++++++++++++++++++++++

You can also use the ``Makefile`` to start and stop the service:

.. code-block:: console

  $ make start
  $ make status
  $ tail -f pywps.log
  $ make stop


Run Albatross as Docker container
---------------------------------

You can also run Albatross as a Docker container.

.. warning::

  TODO: Describe Docker container support.

Use Ansible to deploy Albatross on your System
----------------------------------------------

Use the `Ansible playbook`_ for PyWPS to deploy Albatross on your system.
The Linux x86_64 deployment environment is locked in ``linux-64.spec``. It
includes the Conda runtime dependencies and the playbook's Gunicorn, gevent,
psycopg2 2.9.12, DRMAA 0.7.9, dill and test packages.

Set these inventory variables (including any overrides in ``etc/albatross_dkrz.yml``):

.. code-block:: yaml

   conda_env_use_spec: true
   conda_env_spec_file: linux-64.spec

The old ``spec-list.txt`` has been replaced. Run a full environment deployment
so the updated packages are installed, rather than only updating application code.

To install the same environment manually on Linux x86_64:

.. code-block:: console

   $ conda create -n albatross --file linux-64.spec
   $ conda activate albatross
   $ python -m pip install .

Basemap 1.4.1, its data package and the compatible PyProj 3.6 wheel are
installed by ``pip install .`` through ``requirements.txt``. Basemap and its
data are not included in the Conda spec. Pip replaces the Conda PyProj package
with the version required by Basemap. The pip wheel
avoids the old Conda GEOS pin that conflicts with the deployment's GDAL stack.
``environment.yml`` installs Basemap through its pip section as well.

For other platforms, use ``environment.yml``. Both dependency definitions use
Python 3.11 and PyWPS 4.7, preserving NumPy 1.x and Basemap 1.4 compatibility.


.. _Ansible playbook: http://ansible-wps-playbook.readthedocs.io/en/latest/index.html
