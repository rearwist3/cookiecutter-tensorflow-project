import tensorflow as tf


class BaseModel(tf.keras.models.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def build_graph(self, x_shape):
        input_shape_nobatch = x_shape[1:]
        self.build(x_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)

    def call(self, x):
        raise NotImplementedError