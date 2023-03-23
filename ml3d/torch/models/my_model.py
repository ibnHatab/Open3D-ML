import torch

# use relative import for all imports within ml3d.
from .base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, name="MyModel"):
        super().__init__(name=name)
        # network definition ...

    def forward(self, inputs):
        pass
        # inference code ...

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg_pipeline.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs):
        labels = inputs['data'].labels # processed data from model.preprocess and/or model.transform.

        # Loss is an object of type SemSegLoss. Any new loss can be added to `ml3d/{tf, torch}/modules/semseg_loss.py`
        loss = Loss.weighted_CrossEntropyLoss(results, labels)
        results, labels = Loss.filter_valid_label(results, labels) # remove ignored indices if present.
        return loss, labels, results

    def preprocess(self, data, attr):
        return data