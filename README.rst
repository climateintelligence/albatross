=========
Albatross
=========

.. image:: https://readthedocs.org/projects/albatross/badge/?version=latest
    :target: https://albatross.readthedocs.io/en/latest/?version=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/climateintelligence/albatross?cacheSeconds=300
   :target: https://github.com/climateintelligence/albatross/blob/main/LICENSE
   :alt: GitHub license

ALBATROSS AS A CLIMATE SERVICE
==============================================================================

**Albatross: A Climate Service for Seasonal Forecasting**

  *Albatross is a bird designed for seasonal forecasting of hydroclimatic variables.*

Albatross is deployed as a **Web Processing Service (WPS)**, providing a robust, accessible interface for climate analysis. It produces deterministic forecasts of seasonal hydroclimatic variables (e.g., precipitation, temperature, streamflows,...) for any given location by leveraging global teleconnections.

It is built upon the Niño Index Phase Analysis (NIPA) framework by Zimmerman et al. (2016).

**⚡ WPS Interface Overview**

The Albatross service exposes a single core process, typically named **Drought**, which requires minimal, intuitive inputs to perform complex seasonal forecasts.

**Core Inputs for the Service:**

The service is designed to work with **only four mandatory user inputs** to define your forecasting problem:

1.  **Input File:** The historical time series data (local file or URL).
2.  **Start year / End year:** Define the training/forecast period.
3.  **Target Season:** Define the season of interest (e.g., JFM, DJF).
4.  **Phase mode:** Define the level of El Niño/La Niña phase analysis.

**🚀 Operational Forecasting Modes**

Albatross supports two distinct operational modes, governed by the ``End year`` parameter, allowing for seamless transition from model verification to real-time prediction.

**🗓 Hindcast (Model Calibration and Verification)**

This mode is used for **training, calibrating, and verifying** the predictive skill of the NIPA model using historical data.

* **Goal:** To assess how well the model *would have* performed historically.
* **Availability:** Set the ``End year`` parameter to a year **prior to the current year** (e.g., 2016).
* **Output:** A time series of historical hindcast predictions, suitable for calculating model skill scores (R-squared, correlation, RMSE, etc.).

**🔮 Operational Forecast (New Feature!)**

This new mode enables the generation of predictions for the current or upcoming season using the most recent available climate index data.

* **Goal:** To produce a **single, actionable prediction** for a future target season.
* **Availability:** Set the ``End year`` parameter to the **current year** (2026) or a **future year** (e.g., 2027).
* **Process:** The model is trained on historical data, and then uses the current month's teleconnection index value (e.g., the latest available ONI) to generate the forecast value.
* **Output:** A single point prediction for the specified target season and year.

**🔧 Detailed Input Requirements**

To run the WPS, you must provide the following parameters:

**📄 Input File Format**

-   A plain-text file containing monthly hydroclimatic data (e.g., precipitation, temperature, etc.) for a single location.
-   The file must be structured as follows:
    -   The first line must contain the title (e.g., E-OBS\_precipitation).
    -   The second line must include two numbers: the starting year and ending year.
    -   The following lines must contain monthly values in row-wise format, ordered by year (12 values per year).

| The file must be structured as monthly values in row-wise format, ordered by year (12 values per year).
| **Example file:** `APGD_Como_ppt.txt <https://github.com/climateintelligence/albatross/blob/main/albatross/data/APGD_Como_ppt.txt>`_
| (You can browse from your computer or provide a direct URL, such as a raw GitHub link)

**🧾 Additional Parameters**

+----------------------+----------------------------------------------------------------------------------+
| Parameter            | Description                                                                      |
+======================+==================================================================================+
| Start year           | First year of the time series (e.g., 1950)                                       |
+----------------------+----------------------------------------------------------------------------------+
| End year             | Last year of the time series for calibration/forecast (e.g., 2023)               |
+----------------------+----------------------------------------------------------------------------------+
| Target Season        | Numeric code for the target season: 1 = JFM, ..., 12 = DJF                       |
+----------------------+----------------------------------------------------------------------------------+
| Phase mode (int)     | Whether to apply NIPA separately for El Niño and La Niña phases (1 or 2)         |
+----------------------+----------------------------------------------------------------------------------+


TECHNICAL & DEVELOPMENT DETAILS
==============================================================================

🔍 What NIPA Does (The Method)
-------------------------------

NIPA classifies historical climate data (e.g., Sea Surface Temperature, SST) according to the phases of El Niño, typically using the Oceanic Niño Index (ONI). For each phase (e.g., El Niño, La Niña, Neutral), NIPA:

-   Identifies relevant spatial patterns in SST fields via linear correlation masks
-   Applies Principal Component Analysis (PCA) to reduce dimensionality
-   Calibrates a linear regression model on the first principal component to predict hydroclimatic outcomes

✨ What’s New in Albatross (Technical Innovations)
--------------------------------------------------

Albatross extends the original NIPA framework with several innovations:

-   🌍 New Teleconnection Indices: In addition to ONI, Albatross incorporates other teleconnections like the North Atlantic Oscillation (NAO).
-   📉 Dynamic Dimensionality Reduction: The number of PCA components retained is dynamically chosen based on the explained variance.
-   🧠 Modular Architecture: Built to easily accommodate new climate indices, regions, and variables.

🔭 What’s Next
--------------

We’re actively working on expanding Albatross with more powerful and flexible tools:

-   ➕ More Indices: Planned integration of additional teleconnections (e.g., Pacific Decadal Oscillation, Arctic Oscillation).
-   🔁 Nonlinear Relationships: Introduction of Mutual Information and other nonlinear dependency measures.
-   🤖 Machine Learning Models: Future versions will explore nonlinear models, including Random Forests, and other ML tools.

**🚀 Local Installation (For Developers/Advanced Users)**

To run Albatross locally, contribute to the project, or deploy the PyWPS service on your own server, you must install the package from the source repository.

.. code-block:: bash

    # 1. Clone the repository
    git clone https://github.com/climateintelligence/albatross.git
    cd albatross

    # 2. Install in editable/development mode
    pip install -e .

    # Note: Ensure you have a working Python environment (3.x) and pip installed.


📘 Citation
-----------

-   **Zimmerman, B. G., D. J. Vimont, and P. J. Block (2016)**, Utilizing the state of ENSO as a means for season-ahead predictor selection, Water Resour. Res., 52, 3761–3774, `doi:10.1002/2015WR017644 <https://doi.org/10.1002/2015WR017644>`_.

Documentation
-------------

Learn more about Albatross in its official documentation at
https://albatross.readthedocs.io.

Submit bug reports, questions and feature requests at
https://github.com/climateintelligence/albatross/issues

Contributing
------------

You can find information about contributing in our `Developer Guide`_.

Please use bumpversion_ to release a new version.

License
-------

* Free software: Apache Software License 2.0
* Documentation: https://albatross.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `bird-house/cookiecutter-birdhouse`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`bird-house/cookiecutter-birdhouse`: https://github.com/bird-house/cookiecutter-birdhouse
.. _`Developer Guide`: https://albatross.readthedocs.io/en/latest/dev_guide.html
.. _bumpversion: https://albatross.readthedocs.io/en/latest/dev_guide.html#bump-a-new-version
