# Delivery report

Per-file sources: PROVENANCE.json. Verification: VALIDATION.md.

Non-algorithm changes: portable CLI paths and imports, isolated copied data/output directories, CSV reporting, documentation and packaging. The PRN module is named normalization.py because PRN is a reserved Windows device name and Git could not index prn.py. No algorithm was changed.

## Tree
```text
.gitignore
CITATION.bib
LICENSE
PROVENANCE.json
pyproject.toml
README.md
requirements.txt
THIRD_PARTY_NOTICES.md
VALIDATION.md
assets/method.png
configs/gain_weights.example.csv
configs/main.yaml
configs/methods.yaml
configs/screen.yaml
licenses/BeerLaNet-MIT.txt
licenses/MixStyle-MIT.txt
prdg/dsu.py
prdg/eps.py
prdg/evaluation.py
prdg/io.py
prdg/normalization.py
prdg/training.py
prdg/__init__.py
scripts/eps_sensitivity.py
scripts/evaluate.py
scripts/gain_robustness.py
scripts/normalize_dataset.py
scripts/prepare_txlpbc_lodo.py
scripts/reference_ablation.py
scripts/stats.py
scripts/train.py
scripts/__init__.py
tables/comparison_40_mean_sd.csv
tables/comparison_40_raw.csv
tables/main_120_mean_sd.csv
tables/main_120_raw.csv
tests/baseline_equivalence.json
tests/baseline_evaluation_attempt.json
tests/check_splits.py
tests/compare_original.py
tests/environment.json
tests/evaluation_equivalence.json
tests/pixel_equivalence.json
tests/smoke.json
tests/split_equivalence.json
tests/test_modules.py
prdg/baselines/beerlanet.py
prdg/baselines/mixstyle.py
prdg/baselines/references.py
prdg/baselines/stains.py
prdg/baselines/vahadane.py
prdg/baselines/__init__.py
```

Updated 2026-11-04: AGPL-3.0-only, paper-based independent DSU implementation, and clean-environment dependency/test validation. See VALIDATION.md. No data, weights, caches or logs are committed; the local .venv is ignored.
