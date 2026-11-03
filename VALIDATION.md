# Validation and release limitations

Original project files were only read. Runtime copies, caches, training outputs and weights were kept outside this repository. No remote was created.

| Check | Result |
|---|---|
| EPS/PRN, 20 real images (5 per source) | PASS, mask equality and max absolute output difference = 0 |
| DSU/MixStyle, seeded train/eval behavior | PASS, exact tensor equality |
| Other reference/stain functions | PASS, 40 equivalence checks |
| Four fold membership and order | PASS, exact agreement with original manifests |
| Unit checks | PASS, 3 tests |
| Existing 120-epoch BCCD PRN+DSU seed42 evaluation | PASS, 0.694363565979154, absolute difference 0 |
| Earlier historical Baseline evaluation attempt | FAIL strict tolerance, difference 0.0001055824587175; retained, cause not established |
| Full BCCD PRN+DSU one-epoch smoke | PASS, preprocessing, training, best checkpoint and held-out evaluation completed |
| Clean environment installation | NOT validated; pip check reports OpenCV/NumPy dependency conflict |

The one-epoch held-out result is only a software smoke check, not a scientific result. JSON evidence is in tests/. Final held-out testing was not used to change algorithm parameters.

## Before public release

1. Resolve the recorded environment dependency conflict: OpenCV 5.0.0.93 declares NumPy >=2, while the actual environment uses 1.26.4. No dependency was upgraded or downgraded. Existing runtime checks passed, but a fresh install is not proven.
2. Confirm permission/license for the DSU-derived implementation; no explicit upstream LICENSE was found. The repository MIT license does not override third-party terms. Ultralytics remains subject to its own license.
3. Supply publication authors/DOI in the citation placeholder and confirm permission to redistribute the supplied method-figure image examples.
4. Table evidence paths are relative to the original experiment archive; that private archive and its checkpoints are intentionally not distributed.

No full scientific training was repeated. BeerLaNet requires the separately downloaded pinned official core. No full new BeerLaNet training was claimed.
