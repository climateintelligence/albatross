# Data shipped with Albatross

## Target series of the case studies

These are the target files used in the paper. They also serve as templates for the
format of the **Target variable** input.

| File | Content | Source |
|---|---|---|
| `Java_saobs.txt` | Monthly area-mean precipitation over the island of Java (mm/month), 1981-2010 | SA-OBS v2.0 (BMKG / KNMI) |
| `APGD_Como_ppt.txt` | Monthly area-mean precipitation over the Lake Como basin closed at the Olginate dam, 1971-2019 | APGD v2.0 (MeteoSwiss) |

The series are derived from the original datasets and are subject to the terms of use of
their providers.

## Target file format

Plain text:

1. first line: free-text description;
2. second line: first and last year of the record, separated by a space;
3. one line per year with the twelve monthly values, January to December, separated by
   spaces or tabs.

Incomplete years are completed with `NaN` or a missing-value code (<= -99). Seasons with
missing months are excluded from the analysis and listed in the run configuration.

## Climate indices

The other files are monthly climate indices used for phase conditioning, in the same
format: `NINO 3.4`, `NINO 12` (Niño 1+2), `SOI`, `DMI`, `NAO`, `AO`, `PNA`, `WP`,
`EAWR`, `SCA`, `QBO`. They are taken from NOAA (PSL and CPC). `PDO.txt` and `ONI.txt` are not selectable
in the service.
