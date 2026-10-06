# Open-Meteo Hourly Temperature, Philippine Cities (tutorial sample)

## Summary

`openmeteo_ph_hourly_temperature.csv` is the notebook's default sample: hourly air temperature at 2 m for
Manila, Cebu and Davao over 14 days, 2026-08-26 00:00 to 2026-09-08 23:00 UTC — 3 series × 336 hours,
1,008 rows, long format (`series_id,timestamp,target`). It is **real weather**, so the foundation model
has no guaranteed advantage over a seasonal-naive baseline, and the notebook records which one wins.

The notebook fetches **this committed copy** at run time from a commit-pinned raw URL of this repository
(the commit that added the file):

```text
https://raw.githubusercontent.com/kurtvalcorza/timesfm-forecasting-pipeline/e47259ae75b93e7611c67e13a14d6607997c38f3/examples/sample-data/openmeteo_ph_hourly_temperature.csv
```

SHA-256 `74163ee609cda87869b7c13f4c2aa59343f94b4ba42d2e57331034902fe04f1a`, 31,275 bytes (`SHA256SUMS`).
The file is byte-identical to the copy first fetched from the sibling repository
`kurtvalcorza/chronos-2-forecasting-pipeline` (verified by fetching that repository's pinned URL on
2026-10-05). The notebook refuses a size or digest mismatch before reading the file, and the tests assert
the committed copy, the template constants and `SHA256SUMS` agree.

## Source and licence

- **Provider:** [Open-Meteo](https://open-meteo.com/) historical weather API
  (`https://archive-api.open-meteo.com/v1/archive`), reanalysis estimates (ECMWF ERA5 among the sources)
  for model grid cells several kilometres across — not station readings.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: *Weather data by
  [Open-Meteo.com](https://open-meteo.com/)*. This file is third-party data and is **not** covered by the
  repository's Apache-2.0 licence.
- **Provenance:** fetched on 2026-09-26 by the sibling repository
  `kurtvalcorza/chronos-2-forecasting-pipeline` (`examples/byod-data/open-meteo-ph-temperature/`), whose
  dataset card records the exact API request, the grid cells returned, and that the fetch applied no
  imputation, resampling or scaling. A second fetch on that date matched byte for byte; reanalysis providers
  may revise values later, which is why the notebook pins the commit and the digest.
- **Personal data:** none.

## Why this series

It is Philippine-relevant, reachable from a stable commit-pinned URL, small (31 KB), hourly with a clear
daily cycle (so the seasonal-naive baseline is a real contender), and its window (2026-08-26 to
2026-09-08) lies after every pretraining cut-off the pinned checkpoint's model card states (Wikipedia
pageviews to November 2023, Google Trends to the end of 2022, GiftEvalPretrain, synthetic data). The
checkpoint's Hub revision date (2026-09-02) overlaps the window, so these exact values are unlikely, not
impossible, to be in its pretraining data; weather from the same sources for earlier dates may be.

## Limits

Three grid cells and two weeks in the southwest-monsoon season cannot rank forecasting models, show
seasonal or year-to-year variation, or establish calibration. The notebook's metrics on it are
tutorial evidence (`sample-sanity`), not a benchmark.
