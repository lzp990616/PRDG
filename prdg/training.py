"""Model wiring; transformations occur after the original backbone block."""
from ultralytics.models.yolo.detect import DetectionTrainer
from .dsu import ShallowDSU
from .baselines.mixstyle import ShallowMixStyle

class DSUTrainer(DetectionTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg,weights,verbose)
        for index in (2,4):
            assert type(model.model[index]).__name__ in ('C3k2','C2f')
            model.model[index]=ShallowDSU(model.model[index])
        return model

class MixStyleTrainer(DetectionTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg,weights,verbose)
        for index in (2,4):
            assert type(model.model[index]).__name__ in ('C3k2','C2f')
            model.model[index]=ShallowMixStyle(model.model[index])
        return model

class BeerTrainer(DSUTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        from .baselines.beerlanet import BeerInputStem
        model=super().get_model(cfg,weights,verbose)
        model.model[0]=BeerInputStem(model.model[0])
        return model

def trainer_for(method):
    if 'BeerLaNet' in method:return BeerTrainer
    if method=='MixStyle':return MixStyleTrainer
    if 'DSU' in method or method in ('B0','B1','B2','B3','B4','B5','B6'):return DSUTrainer
    return DetectionTrainer

def disable_oom_retry(trainer):
    # Do not silently change batch size after an out-of-memory error.
    trainer._oom_retries=3
