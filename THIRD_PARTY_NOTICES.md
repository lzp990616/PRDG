# Third-party notices

PRDG is licensed under AGPL-3.0-only. Ultralytics remains under its own AGPL-3.0 terms (or a separately obtained license); third-party components retain their notices and licenses.

* BeerLaNet: https://github.com/xutianyue/BeerLaNet, commit `510b4328ce6de975a2d87643ad7ba940979c0123`, MIT (notice in licenses/). Core source is deliberately not vendored. Obtain that commit separately, set `BEERLANET_SRC` to its `src` directory. The loader checks the published core SHA-256. The local adapter is 8-channel density -> trainable 1x1 convolution (8->3) -> detector, after augmentation; 53 added parameters. r=8, n_iter=10, unit_norm_S=True, calc_tau=True, learn_S_init=True. Local core/adapter math is FP32, detector dtype is restored at the output.
* MixStyle: implementation adapted in the original project from Dassl, MIT; retained notice in licenses/ and attribution in source.
* DSU: **independent implementation from the paper**, Li et al., "Uncertainty Modeling for Out-of-Distribution Generalization", ICLR 2022, Eqs. (1)-(7), https://arxiv.org/abs/2202.03958. No upstream DSU source is bundled. Unbiased variances, epsilon, random-number order and singleton behavior follow the recorded experiment protocol and are tested for exact compatibility. This describes implementation provenance, not a separate legal clearance.
* Stain methods: original project NumPy/OpenCV implementations. Vahadane is a sparse-NMF backend, not the SPAMS reference implementation; Macenko and Reinhard are the exact local paper implementations.
* TXL-PBC is separately obtained, not bundled. The method diagram contains published-example cells; confirm figure/image reuse rights before public release.
