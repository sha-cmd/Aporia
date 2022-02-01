import tensorflow as tf


class BalancedCrossEntropy(tf.keras.metrics.Metric):
    """Balanced cross entropy (BCE) is similar to WCE.
    The only difference is that we weight also
    the negative examples."""
    def __init__(self, beta, **kwargs):
        super(BalancedCrossEntropy, self).__init__(name='bce')
        self.beta = beta

    def get_config(self):
        config = super(BalancedCrossEntropy, self).get_config()
        config['beta'] = self.beta
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def update_state(self, y_true, y_pred, sample_weight=None):
        weight_a = self.beta * tf.cast(y_true, tf.float32)
        weight_b = (1 - self.beta) * tf.cast(1 - y_true, tf.float32)
        self.res = (tf.math.log1p(tf.exp(-tf.abs(y_pred)))
                    + tf.nn.relu(-y_pred)) * (weight_a + weight_b) \
                   + y_pred * weight_b

    def result(self):
        return tf.reduce_mean(self.res)
