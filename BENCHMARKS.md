# Reproduction and benchmark scope

The corrected engine-disjoint training/model-selection path now exports its fitted model, feature order, policy threshold and test predictions. Reloaded predictions match exactly; the retained endpoint MAE remains**13.4434 cycles**.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Existing focused regression suite | None |
| `full-benchmark` | Engine-disjoint fitted model selection and censored historical test replay; model persistence/reload parity | None |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

FD001 is a previously explored simulated-turbofan cohort with right-censored test trajectories. Model accuracy and replay outcomes do not demonstrate real-fleet intervention savings.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Recorded command definitions

### checks

```text
{python} -m pytest -q tests
```

### full-benchmark

```text
{python} validate_release.py
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
