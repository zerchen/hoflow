import torch.nn as nn
import torch


class ModelWrapper(nn.Module):
    def __init__(self, model, metas):
        super().__init__()
        self.model = model.module
        self.metas = metas

    def forward(self, x):
        inputs = dict()
        inputs["img"] = x

        preds = self.model(inputs=inputs, targets=None, metas=self.metas, mode="test")
        return preds
