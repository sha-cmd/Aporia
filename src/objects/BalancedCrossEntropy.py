import tensorflow as tf
import yaml


def BalancedCrossEntropy(y_true, y_pred):
    """Balanced cross entropy (BCE) is similar to WCE.
        The only difference is that we weight also
        the negative examples."""
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        bce_beta = float(params['constants']['wce_beta'])
    weight_a = bce_beta * tf.cast(y_true, tf.float32)
    weight_b = (1 - bce_beta) * tf.cast(1 - y_true, tf.float32)
    res = (tf.math.log1p(tf.exp(-tf.abs(y_pred)))
                + tf.nn.relu(-y_pred)) * (weight_a + weight_b) \
               + y_pred * weight_b
    if res is not None:
        return tf.reduce_mean(res)
    return
