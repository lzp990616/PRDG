# Third-party notices

The MIT license covers original PRDG packaging and project code only. Dependencies retain their own licenses; in particular Ultralytics is AGPL-3.0 (or separately licensed), not MIT. This repository does not relicense Ultralytics.

* BeerLaNet: https://github.com/xutianyue/BeerLaNet, commit `510b4328ce6de975a2d87643ad7ba940979c0123`, MIT (notice in licenses/). Core source is deliberately not vendored. Obtain that commit separately, set `BEERLANET_SRC` to its `src` directory. The loader checks the published core SHA-256. The local adapter is 8-channel density -> trainable 1x1 convolution (8->3) -> detector, after augmentation; 53 added parameters. r=8, n_iter=10, unit_norm_S=True, calc_tau=True, learn_S_init=True. Local core/adapter math is FP32, detector dtype is restored at the output.
* MixStyle: implementation adapted in the original project from Dassl, MIT; retained notice in licenses/ and attribution in source.
* DSU: original project implementation of the algorithm from https://github.com/lixiaotong97/DSU / ICLR 2022. An explicit license for the upstream DSU core could not be verified. Before public distribution, confirm the provenance/redistribution permission of the original project's adaptation. Do not assume the MIT file in this repository grants rights to third-party code. The numerical implementation was retained for reproducibility, not rewritten to evade this uncertainty.
* Stain methods: original project NumPy/OpenCV implementations. Vahadane is a sparse-NMF backend, not the SPAMS reference implementation; Macenko and Reinhard are the exact local paper implementations.
* TXL-PBC is separately obtained, not bundled. The method diagram contains published-example cells; confirm figure/image reuse rights before public release.
