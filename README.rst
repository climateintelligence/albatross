=========
Albatross
=========

.. image:: https://readthedocs.org/projects/clint-albatross/badge/?version=latest
    :target: https://clint-albatross.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/climateintelligence/albatross?cacheSeconds=300
   :target: https://github.com/climateintelligence/albatross/blob/main/LICENSE
   :alt: GitHub license

**Albatross: a climate service for seasonal hydroclimatic forecasting from teleconnections**

  *Albatross is a bird designed for seasonal forecasting of hydroclimatic variables.*

Albatross is an open-source climate service implemented as an OGC **Web Processing
Service (WPS)** with PyWPS, following the Birdhouse framework. The user supplies the
monthly time series of a hydroclimatic variable (e.g., basin precipitation or reservoir
inflow) and a few configuration choices. Albatross retrieves global predictor fields and
climate indices, and returns:

- **out-of-sample hindcasts** and their **skill relative to climatology**, obtained by
  leave-one-out cross-validation in which every data-dependent step is re-estimated
  without the held-out year;
- **maps of the predictor regions** used by the model, to check whether the skill rests
  on a documented teleconnection;
- optionally, a **forecast** for a given year, always returned together with the skill
  of the model that produced it;
- the full configuration and the **calibrated model**, which can be re-applied offline.

The model currently implemented is a generalized version of the Niño Index Phase
Analysis (NIPA; Zimmerman et al., 2016). Albatross is designed to host further
teleconnection-based models under the same data, inputs and validation protocol.

The service is deployed in the CLINT Climate Resilience Information System at DKRZ and
can be accessed through the Phoenix web interface at https://clint.dkrz.de.


Using the service
=================

The service exposes one process, ``drought``.

Inputs
------

Only ``target`` and ``month`` are mandatory; all other inputs have defaults.

+-------------------+-----------+------------------------------------------------------------------+
| Input             | Default   | Description                                                      |
+===================+===========+==================================================================+
| ``target``        | (required)| Monthly time series of the target variable: an uploaded file or  |
|                   |           | a URL (e.g., a raw GitHub link). Format below.                   |
+-------------------+-----------+------------------------------------------------------------------+
| ``month``         | (required)| Target season as consecutive months, e.g. ``4,5,6`` (AMJ) or     |
|                   |           | ``11,12,13,14,15`` (NDJFM). It may start in any month and extend |
|                   |           | at most to March of the following year (``15``).                 |
+-------------------+-----------+------------------------------------------------------------------+
| ``aggregation``   | ``mean``  | How the monthly values are aggregated over the season:           |
|                   |           | ``mean`` (e.g., average inflow) or ``sum`` (e.g., precipitation  |
|                   |           | totals).                                                         |
+-------------------+-----------+------------------------------------------------------------------+
| ``start_year``,   | longest   | Analysis period. By default, the longest period covered by the   |
| ``end_year``      | common    | target, the predictor field and, if used, the climate index.     |
|                   | period    |                                                                  |
+-------------------+-----------+------------------------------------------------------------------+
| ``glo_var_name``  | ``sst``   | Predictor field: ``sst`` (ERSST v5), ``slp`` or ``z500``         |
|                   |           | (NCEP–NCAR Reanalysis 1).                                        |
+-------------------+-----------+------------------------------------------------------------------+
| ``phase_mode``    | ``1``     | *NIPA-specific.* Number of phases: ``1`` = all years, no climate |
|                   |           | index; ``2``, ``3``, ``4`` = years split at the median, terciles |
|                   |           | or quartiles of the climate index.                               |
+-------------------+-----------+------------------------------------------------------------------+
| ``indicator``     | none      | *NIPA-specific, required if* ``phase_mode`` *> 1.* Climate index:|
|                   |           | ``NINO 3.4``,  ``NINO 12``, ``ONI``,                             |
|                   |           | ``SOI``, ``DMI``, ``NAO``, ``AO``, ``PNA``, ``WP``, ``EAWR``,    |
|                   |           | ``SCA``, ``QBO``.                                                |
+-------------------+-----------+------------------------------------------------------------------+
| ``forecast_year`` | none      | Year of the target season to forecast. If omitted, only          |
|                   |           | hindcasts and skill are returned.                                |
+-------------------+-----------+------------------------------------------------------------------+

The predictor field and the climate index are averaged over the **three months
preceding the target season** (e.g., JFM for an AMJ target), so forecasts are issued at
the start of the target season.

Target file format
------------------

A plain-text file with:

1. a first line with a free-text description;
2. a second line with the first and last year of the record (e.g., ``1971 2019``);
3. the monthly values, **one year per row, January to December**, starting in January of
   the first year, with no year column.

Values must be in absolute terms (not anomalies). Missing months must be written as
``NaN`` or as a missing-value code (``-99``, ``-99.9``, ``-999``, ...), never omitted,
since the dates are inferred from the position of each value. Years whose target season
contains missing months are excluded from the analysis and listed in the outputs.

Example files are available in
`albatross/data <https://github.com/climateintelligence/albatross/tree/main/albatross/data>`_.

Outputs
-------

Each execution returns an archive (``forecast_bundle``) containing all outputs; the main
products are also returned individually as WPS outputs.

- ``operational_forecast``: the forecast for ``forecast_year`` (if requested), with its
  trend and teleconnection components, the phase and index value used, and the
  leave-one-out skill of that phase.
- ``forecast_file``: the observed target and the leave-one-out hindcast for each year.
- ``metrics_table``: MAE, RMSE, bias, R², Pearson correlation (one-sided p-value) and skill
  scores relative to the leave-one-out climatology, for each phase and, with more than one
  phase, for all phases combined.
- ``scatter_plot``: observed values against leave-one-out hindcasts, one panel per phase.
- ``glo_var_map``: grid cells of the predictor field retained in at least half of the
  leave-one-out folds, colored by their correlation with the target.
- In the archive only: the run configuration (``training_config.json``), the phase
  thresholds, the out-of-sample principal components and coefficients of every fold, and
  the operational model of each phase (``final_model_<phase>.npz``).


Method
======

For each phase, the generalized NIPA model performs the following steps:

1. **Climate state detection** (if ``phase_mode`` > 1): years are split into phases of
   equal size according to the climate index.
2. **Trend removal**: a linear trend is removed from the target and from each grid cell
   of the predictor field.
3. **Screening**: grid cells whose correlation with the target is significant at the 5%
   level are retained; cells poleward of 60° are excluded, and spatially isolated cells
   are removed.
4. **Dimensionality reduction**: the retained cells, weighted by the cosine of latitude,
   are summarized by their leading principal component.
5. **Regression**: the detrended target is regressed on the leading principal component
   by ordinary least squares.

**Validation.** Hindcasts are obtained by leave-one-out cross-validation within each
phase. Steps 2–5 are re-estimated for every held-out year using only the remaining
years, so the held-out target influences neither the retained cells nor the trend, the
principal component or the regression. Selecting the predictor region on the full
record and cross-validating only the regression would overestimate skill.

**Forecast.** The operational model of each phase applies the same steps to all years
of the phase. The reported skill is that of the leave-one-out hindcasts, which evaluate
exactly this procedure on years excluded from estimation.

**Offline use.** The operational model saved in the archive can be applied without the
WPS with ``albatross.utils.apply_final_model(npz_path, field_2d, year)``, where
``field_2d`` is the raw predictor field averaged over the same three months, on the same
grid.


Installation (developers)
=========================

.. code-block:: bash

    git clone https://github.com/climateintelligence/albatross.git
    cd albatross
    pip install -e .

    # start the WPS service locally (see the Developer Guide for options)
    albatross start

The service is then available at ``http://localhost:<port>/wps``; the port is printed at
start-up and set in the service configuration.

Predictor fields are downloaded from the NOAA Physical Sciences Laboratory on first use,
cached locally, and refreshed when a request needs months not yet in the cached copy.
The following environment variables control this behaviour:

+------------------------------+---------------------------------------------------------------+
| Variable                     | Meaning                                                       |
+==============================+===============================================================+
| ``ALBATROSS_DATA_DIR``       | Folder of the cached predictor files (default ``~/data``).    |
+------------------------------+---------------------------------------------------------------+
| ``ALBATROSS_SST_NC``,        | Path of a single predictor file (SST, SLP, geopotential       |
| ``ALBATROSS_SLP_NC``,        | height, orography), overriding ``ALBATROSS_DATA_DIR``.        |
| ``ALBATROSS_HGT_NC``,        |                                                               |
| ``ALBATROSS_OROG_NC``        |                                                               |
+------------------------------+---------------------------------------------------------------+
| ``ALBATROSS_REFRESH_HOURS``  | Minimum interval between two refreshes of the same file       |
|                              | (default 24).                                                 |
+------------------------------+---------------------------------------------------------------+
| ``ALBATROSS_OUT_DIR``        | Folder where a copy of each run is saved (default             |
|                              | ``~/Desktop``).                                               |
+------------------------------+---------------------------------------------------------------+

Production of the NCEP–NCAR Reanalysis 1 ended in March 2026: SLP and Z500 are not
available after that date.


Roadmap
=======

- Further teleconnection-based models selectable as an input, such as models combining
  remote and local predictors and causal precursor discovery, under the same validation
  protocol.
- Within NIPA: nonlinear identification of predictor patterns, machine-learning
  regressors, more than one principal component (with nested validation), and phase
  conditioning on the joint state of several indices.
- Prediction intervals from the leave-one-out residuals, and longer lead times.


Citation
========

If you use Albatross, please cite:

- Palcic, G., et al. Albatross: A Modular Climate Service for Seasonal Hydroclimatic
  Forecasting from Teleconnections. *Climate Services* (submitted).
- Zimmerman, B. G., D. J. Vimont, and P. J. Block (2016), Utilizing the state of ENSO as a
  means for season-ahead predictor selection, *Water Resour. Res.*, 52, 3761–3774,
  `doi:10.1002/2015WR017644 <https://doi.org/10.1002/2015WR017644>`_.


Documentation and support
=========================

Documentation: https://clint-albatross.readthedocs.io/en/latest/

Bug reports, questions and feature requests:
https://github.com/climateintelligence/albatross/issues

Contributing
------------

See the `Developer Guide`_. Please use bumpversion_ to release a new version.

License
-------

Apache Software License 2.0.

Credits
-------

This package was created with Cookiecutter_ and the `bird-house/cookiecutter-birdhouse`_
project template. Developed within the CLINT project (Horizon 2020, grant agreement
101003876).

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`bird-house/cookiecutter-birdhouse`: https://github.com/bird-house/cookiecutter-birdhouse
.. _`Developer Guide`: https://clint-albatross.readthedocs.io/en/latest/dev_guide.html
.. _bumpversion: https://clint-albatross.readthedocs.io/en/latest/dev_guide.html#bump-a-new-version
