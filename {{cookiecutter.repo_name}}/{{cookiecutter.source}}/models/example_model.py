from base.base_model import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, cfg):
        super(ExampleModel, self).__init__(cfg)
        self.latent_dim = cfg.latent_dim
        self.base_filsize = cfg.base_filsize
        self.depth = cfg.depth

    def call(self, x):
        pass