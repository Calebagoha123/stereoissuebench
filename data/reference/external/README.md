# External demographic reference tables

Per-name demographic distributions joined onto the Arm-B name bank by
`pipeline/build_name_bank.py`. Both are public; download from Harvard Dataverse
(the API needs a browser `User-Agent` to clear the WAF challenge).

## Tzioumis 2018 — name frequency + P(race) (committed: `tzioumis_firstnames.csv`)
- Source: Tzioumis (2018), "Demographic aspects of first names", *Scientific Data*.
- Dataverse DOI: `10.7910/DVN/TYJKEZ` (original file `firstnames.xlsx`, sheet 2).
- Columns: `firstname, obs, pcthispanic, pctwhite, pctblack, pctapi, pctaian, pct2prace`
  (percentages 0–100). Supplies `cov_freq` (= `obs`) and a cross-check P(race|name).
- Coverage of the Tonneau names: ~62% (Black) to ~74% (White).

## Rosenman 2023 — P(race | first name) (NOT committed; gitignored, 7.9 MB)
- Source: Rosenman, Olivella, Imai (2023), "Race and ethnicity data for first,
  middle, and last names", *Scientific Data*.
- Dataverse DOI: `10.7910/DVN/SGKW0K`, file `first_nameRaceProbs.tab` (id 7060179).
- Columns: `name, whi, bla, his, asi, oth` (probabilities 0–1). Supplies
  `cov_p_group` (= `bla`/`whi` for the name's group). **100% coverage** of the
  Tonneau names — this is the primary covariate.

### Re-fetch
```sh
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
curl -sS -A "$UA" -L \
  "https://dataverse.harvard.edu/api/access/datafile/7060179" \
  -o data/reference/external/rosenman_first_nameRaceProbs.tab
```

The built `data/input/names/name_bank.csv` is the committed canonical input, so
these raw tables are only needed to regenerate the bank.
