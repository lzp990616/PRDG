# Validation and release limitations

Original project files were only read. Runtime copies, training outputs and weights were kept outside this repository. The new isolated .venv is ignored by Git and is not distributed.

| Check | Result |
|---|---|
| EPS/PRN, 20 real images (5 per source) | PASS, mask equality and max absolute output difference = 0 |
| DSU/MixStyle, seeded train/eval behavior | PASS, exact tensor equality |
| Other reference/stain functions | PASS, 40 equivalence checks |
| Four fold membership and order | PASS, exact agreement with original manifests |
| Unit checks | PASS, 4 tests |
| Existing 120-epoch BCCD PRN+DSU seed42 evaluation | PASS, 0.694363565979154, absolute difference 0 |
| Earlier historical Baseline evaluation attempt | FAIL strict tolerance, difference 0.0001055824587175; retained, cause not established |
| Full BCCD PRN+DSU one-epoch smoke | PASS, preprocessing, training, best checkpoint and held-out evaluation completed |
| Clean environment installation (2026-11-04) | PASS, isolated Python 3.12.2 venv, OpenCV 4.10.0.84 / NumPy 1.26.4; pip check: No broken requirements found |

The one-epoch held-out result is only a software smoke check, not a scientific result. JSON evidence is in tests/. Final held-out testing was not used to change algorithm parameters.

## Clean-environment verification, 2026-11-04

A new venv was created without system-site-packages, then installed from requirements.txt. No package in the historical experiment environment was changed. All tests used CPU; no training or detector re-evaluation was run for this revision.

- `python -m pip check`: PASS, no broken requirements.
- `python tests/compare_original.py --original <original-project> --out tests/pixel_equivalence_clean.json`: PASS, 20 real images (five per source), EPS/PRN max absolute difference 0, target reduction exact, DSU/MixStyle exact.
- The rewritten DSU additionally passes 30 cases covering three tensor shapes, five seeds, training/eval, singleton batches, exact outputs, input gradients and post-call Torch/NumPy RNG state. Defaults remain p=0.5, eps=1e-6, factor=1 with unbiased variances. The historical environment also passes: tests/dsu_rewrite_historical_environment.json.
- `python tests/check_splits.py --original-splits <original-splits> --prepared <prepared-folds> --out tests/split_equivalence_clean.json`: PASS, all four folds retain identical membership and order. Existing prepared fold copies were checked; data were not regenerated.
- `python -m unittest discover -s tests -p test_modules.py`: PASS, four tests including the <64-pixel masked_mean fallback.

Full installed package versions and command outcomes: tests/clean_environment.json. Original versions remain in tests/environment.json. The earlier detector evaluation and one-epoch smoke above belong to that historical environment, before this rewrite; they are not claimed as repeated in the new environment.

The old OpenCV 5.0.0.93 / NumPy 1.26.4 dependency conflict is resolved in the new environment by the authorized OpenCV 4.10.0.84 pin. DSU is now independently implemented from the paper, with equation citations; AGPL-3.0-only applies to PRDG and BeerLaNet/MixStyle MIT notices remain. This is an implementation/provenance statement, not a legal audit.

The 1260-image metadata has minimum image area 65536 pixels. EPS's >=3% final-mask coverage implies at least 1967 pixels at that size, so the <64 fallback cannot be reached at the experiment resolutions. Tiny arbitrary images can still require the safety fallback.

## Remaining publication details

Supply publication authors/DOI in the citation placeholder and confirm permission to redistribute the supplied method-figure examples. Table evidence paths are relative to the private original archive, which is not distributed. BeerLaNet requires its separately downloaded pinned core. No full scientific training or new BeerLaNet training was repeated.

## Recorded interruptions

The first module-test run failed allocating a 9.16 MiB NumPy bootstrap array; the unchanged four-test suite passed on rerun. An intermediate attempt stopped because the evidence writer refused to overwrite pixel_equivalence_clean.json. No algorithm, bootstrap count or tolerance changed. Both attempts are retained in tests/clean_environment_initial_attempt.json and tests/clean_environment_output_collision.json. The final clean_environment.json records the combined successful checks and this history.
