import tensorflow as tf

def instance_norm2D(x, scope='instance_norm', **kwargs):

    with tf.variable_scope(scope):

        depth = x.get_shape()[3]
        scale = tf.get_variable('scale', [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return scale*normalized + offset


def group_norm2D(x, eps=1e-5, scope='group_norm', **kwargs) :

    with tf.variable_scope(scope) :

        N  = tf.shape(x)[0]
        _, H, W, C = x.get_shape().as_list()
        # G = min(G, C)

        G = kwargs.get('num_groups', max(2, C // 16))  # 16 channels per group gave good results in paper, but at least 2

        x = tf.reshape(x, tf.stack([N, H, W, G, C // G]))
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x


def layer_norm(x,
               gamma=None,
               beta=None,
               axes=(1, 2, 3),
               eps=1e-3,
               scope='layer_norm',
               **kwargs):

    """
    Collect mean and variances on x except the first dimension. And apply normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps)
    :param x: Input variable
    :param gamma: scaling parameter
    :param beta: bias parameter
    :param axes: which axes to collect the statistics over (default is correct for 2D conv)
    :param eps: Denominator bias
    :param name: Name of the layer
    :return: Returns the normalised version of x
    """

    with tf.variable_scope(scope):
        mean, var = tf.nn.moments(x, axes, name='moments', keep_dims=True)
        normed = (x - mean) / tf.sqrt(eps + var)
        if gamma is not None:
          normed *= gamma
        if beta is not None:
          normed += beta
        normed = tf.identity(normed, name='normed')

        return normed



def batch_renorm(x, training, moving_average_decay=0.99, scope='batch_renorm', **kwargs):
    '''
    Batch renormalisation implementation using tf batch normalisation function.
    :param x: Input layer (should be before activation)
    :param name: A name for the computational graph
    :param training: A tf.bool specifying if the layer is executed at training or testing time
    :param moving_average_decay: Moving average decay of data set mean and std
    :return: Batch normalised activation
    '''

    def parametrize_variable(global_step, y_min, y_max, x_min, x_max):
        # Helper function to create a linear increase of a variable from (x_min, y_min) to (x_max, y_max) paramterised
        # by the global number of iterations (global_step).

        # if x < x_min:
        #     return y_min
        # elif x > x_max:
        #     return y_max
        # else:
        #     return (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min

        x = tf.to_float(global_step)

        def f1(): return tf.constant(y_min)

        def f2(): return tf.constant(y_max)

        def f3(): return ((x - x_min) * (y_max - y_min) / (x_max - x_min)) + y_min

        y = tf.case({tf.less(x, x_min): f1,
                     tf.greater(x, x_max): f2},
                    default=f3,
                    exclusive=True)

        return y

        ## End helper function

    rmin = 1.0
    rmax = 3.0

    dmin = 0.0
    dmax = 5.0

    # values /10 from paper because training goes faster for us
    x_min_r = 5000.0 / 10
    x_max_r = 40000.0 / 10

    x_min_d = 5000.0 / 10
    x_max_d = 25000.0 / 10

    global_step = tf.train.get_or_create_global_step()

    clip_r = parametrize_variable(global_step, rmin, rmax, x_min_r, x_max_r)
    clip_d = parametrize_variable(global_step, dmin, dmax, x_min_d, x_max_d)

    tf.summary.scalar('rmax_clip', clip_r)
    tf.summary.scalar('dmax_clip', clip_d)

    with tf.variable_scope(scope):

        h_bn = tf.contrib.layers.batch_norm(inputs=x,
                                            renorm_decay=moving_average_decay,
                                            epsilon=1e-3,
                                            is_training=training,
                                            center=True,
                                            scale=True,
                                            renorm=True,
                                            renorm_clipping={'rmax': clip_r, 'dmax': clip_d})

    return h_bn


def batch_norm(x, training, moving_average_decay=0.99, scope='batch_norm', **kwargs):
    '''
    Wrapper for tensorflows own batch normalisation function. 
    :param x: Input layer (should be before activation)
    :param name: A name for the computational graph
    :param training: A tf.bool specifying if the layer is executed at training or testing time
    :return: Batch normalised activation
    '''

    with tf.variable_scope(scope):

        h_bn = tf.contrib.layers.batch_norm(inputs=x,
                                            decay=moving_average_decay,
                                            epsilon=1e-3,
                                            is_training=training,
                                            center=True,
                                            scale=True)

    return h_bn


def identity(x, **kwargs):
    '''
    Wrapper for tf idenity function, which allows to pass extra arguments that might be needed for other normalisers
    via kwargs.
    '''
    return tf.identity(x)