=========
Albatross
=========

.. image:: https://readthedocs.org/projects/clint-albatross/badge/?version=latest
    :target: https://clint-albatross.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/climateintelligence/albatross?cacheSeconds=300
   :target: https://github.com/climateintelligence/albatross/blob/main/LICENSE
   :alt: GitHub license

Albatross is an open-source climate service for seasonal forecasting of hydroclimatic
variables from teleconnections, implemented as an OGC Web Processing Service (WPS) with
PyWPS. The user supplies the monthly time series of a target variable (e.g. precipitation
over a catchment) and a few configuration choices; Albatross retrieves global predictor
fields and climate indices, returns leave-one-out hindcasts with their skill against
climatology, maps of the predictor regions, and, optionally, a forecast for a given year.

The model is a generalized version of the Niño Index Phase Analysis (NIPA,
Zimmerman et al., 2016).

The service is deployed at DKRZ and can be used through the Phoenix web interface at
https://clint.dkrz.de.

Inputs
======

The inputs are listed in the order of the web interface. Only the first two are mandatory.

+-----------------------------------------------+-----------------------------------------------------------------------------+
| Input                                         | Description                                                                 |
+===============================================+=============================================================================+
| Target variable                               | Plain-text file (upload or URL) with the monthly target series; see below.  |
+-----------------------------------------------+-----------------------------------------------------------------------------+
| Target season                                 | Comma-separated consecutive months, e.g. ``3,4,5``; months of the           |
|                                               | following year are 13-15, e.g. ``11,12,13,14,15`` for November-March.       |
+-----------------------------------------------+-----------------------------------------------------------------------------+
| Temporal aggregation                          | ``mean`` (default) or ``sum`` of the monthly values over the season.        |
+-----------------------------------------------+-----------------------------------------------------------------------------+
| Analysis period: start year / end year        | Optional. Default: the full period covered by target, predictor field and,  |
|                                               | if used, climate index. Years outside this coverage return an error.        |
+-----------------------------------------------+-----------------------------------------------------------------------------+
| Predictor field                               | ``sst`` (default), ``slp`` or ``z500``.                                     |
+-----------------------------------------------+-----------------------------------------------------------------------------+
| Phase conditioning: number of phases          | 1 (default, all years) to 4.                                                |
+-----------------------------------------------+-----------------------------------------------------------------------------+
| Phase conditioning: climate index             | Required with more than one phase; no effect with one phase.                |
+-----------------------------------------------+-----------------------------------------------------------------------------+
| Forecast year                                 | Optional. Year of the target season to forecast.                            |
+-----------------------------------------------+-----------------------------------------------------------------------------+

The predictor field and the climate index are averaged over the three months preceding the
target season.

Target file format
------------------

- first line: free-text description;
- second line: first and last year of the record, separated by a space;
- one line per year with the twelve monthly values, January to December.

Incomplete years are completed with ``NaN`` or a missing-value code (<= -99). The target
files of the two case studies of the paper, ``Java_saobs.txt`` and ``APGD_Como_ppt.txt``,
are in ``albatross/data`` and can be used as templates.

Outputs
=======

+------------------+------------------------+--------------------------------------------------------------------+
| Output           | File                   | Content                                                            |
+==================+========================+====================================================================+
| Forecast         | ``forecast.csv``       | Forecast, its trend and teleconnection components, and the         |
|                  |                        | leave-one-out skill of the model that produced it.                 |
+------------------+------------------------+--------------------------------------------------------------------+
| Metrics table    | ``metrics.csv``        | Errors, correlation and skill scores of the leave-one-out          |
|                  |                        | hindcasts, per phase and for all phases combined.                  |
+------------------+------------------------+--------------------------------------------------------------------+
| Hindcast table   | ``hindcasts.csv``      | Observed value, leave-one-out hindcast and trend component per     |
|                  |                        | year.                                                              |
+------------------+------------------------+--------------------------------------------------------------------+
| Predictor maps   | ``predictor_map.pdf``  | Cells retained in at least half of the folds, one panel per phase. |
+------------------+------------------------+--------------------------------------------------------------------+
| Scatter plots    | ``scatterplot.pdf``    | Observed against hindcast, one panel per phase.                    |
+------------------+------------------------+--------------------------------------------------------------------+
| All outputs      | ``outputs.zip``        | All the above, the target series and a ``model/`` folder with the  |
|                  |                        | run configuration, the operational model of each phase and the     |
|                  |                        | diagnostics of every fold.                                         |
+------------------+------------------------+--------------------------------------------------------------------+

Method
======

For each phase of the climate index, Albatross removes a linear trend from the target and
the predictor field, retains the grid cells significantly correlated with the target
(5% level, with a spatial coherence filter), summarizes them by their first principal
component, and regresses the target on it. All these steps are re-estimated in every
leave-one-out fold, so that the hindcasts are out-of-sample with respect to predictor
selection. Skill is measured against the leave-one-out climatology.

Predictor data
--------------

Predictor fields are downloaded from NOAA PSL on first use and cached locally
(``ALBATROSS_DATA_DIR``, default ``~/data``):

- SST: NOAA ERSST v6;
- SLP and Z500: NCEP/NCAR Reanalysis 1, which is no longer updated after February 2026.
  Forecasts based on SLP or Z500 are therefore limited to target seasons whose predictor
  window ends by February 2026.

Offline forecast
================

``offline_forecast.py`` applies an operational model returned by the service without the
service or the albatross package (dependencies: ``numpy`` and ``netCDF4``). It downloads
the predictor field from NOAA PSL if needed.

With one phase, pass the model from the ``model/`` folder of ``outputs.zip``:

.. code-block:: bash

    python offline_forecast.py --model model/final_model_allyears.npz \
        --predictor sst --target-months 10,11,12 --year 2011 --check forecast.csv

With more than one phase, pass the model folder, the phase thresholds and the climate
index file; the script selects the phase from the index:

.. code-block:: bash

    python offline_forecast.py --model-dir model \
        --thresholds model/phase_thresholds.json --index "albatross/data/NINO 3.4.txt" \
        --predictor sst --target-months 10,11,12 --year 2011

``--check`` compares the result with the ``forecast.csv`` returned by the service.
Run ``python offline_forecast.py --help`` for all options.

Local installation
==================

.. code-block:: bash

    git clone https://github.com/climateintelligence/albatross.git
    cd albatross
    pip install -e .
    albatross start          # WPS at http://localhost:5000/wps

Roadmap
=======

- **Forecast products**: prediction intervals derived from the leave-one-out residuals,
  and longer lead times.
- **Predictor data**: migration of SLP and Z500 from NCEP/NCAR Reanalysis 1, no longer
  updated, to a maintained reanalysis such as ERA5.
- **Within NIPA**: nonlinear methods for identifying predictor patterns,
  machine-learning regressors in place of the linear model, and more than one principal
  component, with a nested validation loop to select their number.
- **Further teleconnection-based models** alongside NIPA, such as models combining remote
  and local predictors (Ionita et al., 2008; Ionita and Nagavciuc, 2020) and causal
  precursor discovery (Di Capua et al., 2019; Vijverberg et al., 2020), sharing the same
  data, inputs and validation protocol, so that their skill can be compared on the same
  target.

References
==========

- Di Capua, G., M. Kretschmer, J. Runge, A. Alessandri, R. V. Donner, B. van den Hurk,
  R. Vellore, R. Krishnan, and D. Coumou (2019), Long-lead statistical forecasts of the
  Indian summer monsoon rainfall based on causal precursors, *Weather Forecast.*, 34,
  1377-1394, `doi:10.1175/WAF-D-19-0002.1 <https://doi.org/10.1175/WAF-D-19-0002.1>`_.
- Ionita, M., G. Lohmann, and N. Rimbu (2008), Prediction of spring Elbe discharge based
  on stable teleconnections with winter global temperature and precipitation,
  *J. Climate*, 21, 6215-6226,
  `doi:10.1175/2008JCLI2248.1 <https://doi.org/10.1175/2008JCLI2248.1>`_.
- Ionita, M., and V. Nagavciuc (2020), Forecasting low flow conditions months in advance
  through teleconnection patterns, with a special focus on summer 2018, *Sci. Rep.*, 10,
  `doi:10.1038/s41598-020-70060-8 <https://doi.org/10.1038/s41598-020-70060-8>`_.
- Vijverberg, S., M. Schmeits, K. van der Wiel, and D. Coumou (2020), Subseasonal
  statistical forecasts of Eastern U.S. hot temperature events, *Mon. Weather Rev.*,
  148, 4799-4822, `doi:10.1175/MWR-D-19-0409.1 <https://doi.org/10.1175/MWR-D-19-0409.1>`_.
- Zimmerman, B. G., D. J. Vimont, and P. J. Block (2016), Utilizing the state of ENSO as a
  means for season-ahead predictor selection, Water Resour. Res., 52, 3761-3774,
  `doi:10.1002/2015WR017644 <https://doi.org/10.1002/2015WR017644>`_.

Documentation
-------------

Learn more about Albatross in its official documentation at
https://clint-albatross.readthedocs.io/en/latest/

Submit bug reports, questions and feature requests at
https://github.com/climateintelligence/albatross/issues

Contributing
------------

You can find information about contributing in our `Developer Guide`_.

Please use bumpversion_ to release a new version.

License
-------

* Free software: Apache Software License 2.0
* Documentation: https://clint-albatross.readthedocs.io/en/latest/


Credits
-------

This package was created with Cookiecutter_ and the `bird-house/cookiecutter-birdhouse`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`bird-house/cookiecutter-birdhouse`: https://github.com/bird-house/cookiecutter-birdhouse
.. _`Developer Guide`: https://clint-albatross.readthedocs.io/en/latest/dev_guide.html
.. _bumpversion: https://clint-albatross.readthedocs.io/en/latest/dev_guide.html#bump-a-new-version
