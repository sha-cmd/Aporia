import tensorflow as tf


class WeightedCrossEntropy(tf.keras.metrics.Metric):
    """Weighted cross entropy (WCE) is a variant of CE where all
     positive examples get weighted by some coefficient. It is used
     in the case of class imbalance. In segmentation, it is often not
     necessary. However, it can be beneficial when the training of
      the neural network is unstable. In classification, it is
       mostly used for multiple classes."""
    def __init__(self, beta, **kwargs):
        super(WeightedCrossEntropy, self).__init__(name='wce')
        self.beta = beta

    def get_config(self):
        config = super(WeightedCrossEntropy, self).get_config()
        config['beta'] = self.beta
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def update_state(self, y_true, y_pred, sample_weight=None):
        weight_a = self.beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)
        self.res = (tf.math.log1p(tf.exp(-tf.abs(y_pred)))
                  + (tf.nn.relu(-y_pred)) * (weight_a + weight_b)) \
                 + (y_pred * weight_b)

    def result(self):
        return tf.reduce_mean(self.res)

