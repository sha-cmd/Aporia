import tensorflow as tf
import yaml


def WeightedCrossEntropy(y_true, y_pred):
    """Weighted cross entropy (WCE) is a variant of CE where all
     positive examples get weighted by some coefficient. It is used
     in the case of class imbalance. In segmentation, it is often not
     necessary. However, it can be beneficial when the training of
      the neural network is unstable. In classification, it is
       mostly used for multiple classes."""
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        wce_beta = float(params['constants']['wce_beta'])
    weight_a = wce_beta * tf.cast(y_true, tf.float32)
    weight_b = 1 - tf.cast(y_true, tf.float32)
    res = (tf.math.log1p(tf.exp(-tf.abs(y_pred)))
                + (tf.nn.relu(-y_pred)) * (weight_a + weight_b)) \
               + (y_pred * weight_b)
    if res is not None:
        return tf.reduce_mean(res)
    return 
