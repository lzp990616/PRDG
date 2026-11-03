import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import unittest
import numpy as np,torch
from prdg.eps import background_mask_v2
from prdg.dsu import DistributionUncertainty
from scripts.eps_sensitivity import variant
from scripts.stats import paired
class Checks(unittest.TestCase):
    def test_eps_default(self):
        for seed in range(5):
            im=np.random.default_rng(seed).integers(0,256,(64,64,3),dtype=np.uint8)
            self.assertTrue(np.array_equal(background_mask_v2(im),variant(im)[0]))
    def test_dsu_singleton_and_eval(self):
        x=torch.randn(1,5,9,9)
        self.assertTrue(torch.equal(DistributionUncertainty(p=1).train()(x),x))
        self.assertTrue(torch.equal(DistributionUncertainty(p=1).eval()(x.repeat(2,1,1,1)),x.repeat(2,1,1,1)))
    def test_paired_constant(self):
        r=paired([.1]*12);self.assertEqual(r['wins'],12);self.assertAlmostEqual(r['delta'],.1)
        self.assertAlmostEqual(r['ci_low'],.1);self.assertAlmostEqual(r['ci_high'],.1)
if __name__=='__main__':unittest.main()
