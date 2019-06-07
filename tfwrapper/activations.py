import tensorflow as tf

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)
