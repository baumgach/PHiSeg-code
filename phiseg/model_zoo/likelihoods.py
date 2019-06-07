import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def betaVAE_bn(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    """
    This is a U-NET like arch with skips before and after latent space and a rather simple decoder
    """

    resolution_levels = kwargs.get('resolution_levels', 5)
    final_kernel_size = [s//(2**(resolution_levels-1)) for s in image_size[0:2]]
    logging.info('@ likelihood final kernel size')
    logging.info(final_kernel_size)

    n0 = kwargs.get('n0', 32)

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        lat_ups = layers.dense_layer(z_list[0], 'z_ups_1', hidden_units=8*n0, normalisation=norm, training=training)
        lat_ups = layers.dense_layer(lat_ups, 'z_ups_2', hidden_units=n0*4*np.prod(final_kernel_size), normalisation=norm, training=training)

        z_reshaped = tf.reshape(lat_ups, tf.stack([-1, final_kernel_size[0], final_kernel_size[1], 4*n0]))

        net = z_reshaped
        for ii in reversed(range(resolution_levels-2)):
            net = layers.transposed_conv2D(net, num_filters=n0*(ii//2+1), name='deconv%d' % ii, normalisation=norm, training=training)

        s = [layers.transposed_conv2D(net, num_filters=n_classes, name='deconv_out_s', activation=tf.identity)]

        return s

#
def unet_T_L(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm,**kwargs):
    """
    This is a U-NET like arch with skips before and after latent space and a rather simple decoder
    """

    n0 = kwargs.get('n0', 32)
    max_channel_power = kwargs.get('max_channel_power', 4)
    max_channels = n0 * 2 ** max_channel_power

    def increase_resolution(x, times, name):

        with tf.variable_scope(name):
            nett = x

            for i in range(times):
                nett = layers.bilinear_upsample2D(nett, 'ups_%d' % i, 2)
                nC = nett.get_shape().as_list()[3]
                nett = layers.conv2D(nett, 'z%d_post' % i, num_filters=min(nC * 2, max_channels), normalisation=norm, training=training)

        return nett

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        resolution_levels = kwargs.get('resolution_levels', 6)
        latent_levels = kwargs.get('latent_levels', 3)

        n0 = kwargs.get('n0', 32)

        post_z = [None] * latent_levels
        post_c = [None] * latent_levels

        s = [None] * latent_levels

        # Generate post_z
        for i in range(latent_levels):
            net = layers.conv2D(z_list[i], 'z%d_post_1' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)
            net = layers.conv2D(net, 'z%d_post_2' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)
            net = increase_resolution(net, resolution_levels - latent_levels, name='preups_%d' % i)

            post_z[i] = net

        # Upstream path
        post_c[latent_levels - 1] = post_z[latent_levels - 1]

        for i in reversed(range(latent_levels - 1)):
            ups_below = layers.bilinear_upsample2D(post_c[i + 1], name='post_z%d_ups' % (i + 1), factor=2)
            ups_below = layers.conv2D(ups_below, 'post_z%d_ups_c' % (i + 1), num_filters=n0 * (i//2+1), normalisation=norm, training=training)

            concat = tf.concat([post_z[i], ups_below], axis=3, name='concat_%d' % i)

            net = layers.conv2D(concat, 'post_c_%d_1' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)
            net = layers.conv2D(net, 'post_c_%d_2' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)

            post_c[i] = net

        # Outputs
        for i in range(latent_levels):
            s_in = layers.conv2D(post_c[i], 'y_lvl%d' % i, num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)
            s[i] = tf.image.resize_images(s_in, image_size[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return s


def unet_T_L_noconcat(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    """
    This is a U-NET like arch with skips before and after latent space and a rather simple decoder
    """

    n0 = kwargs.get('n0', 32)
    max_channel_power = kwargs.get('max_channel_power', 4)
    max_channels = n0 * 2 ** max_channel_power

    def increase_resolution(x, times, name):

        with tf.variable_scope(name):
            nett = x

            for i in range(times):
                nett = layers.bilinear_upsample2D(nett, 'ups_%d' % i, 2)
                nC = nett.get_shape().as_list()[3]
                nett = layers.conv2D(nett, 'z%d_post' % i, num_filters=min(nC * 2, max_channels), normalisation=norm, training=training)

        return nett

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        resolution_levels = kwargs.get('resolution_levels', 6)
        latent_levels = kwargs.get('latent_levels', 3)

        n0 = kwargs.get('n0', 32)

        post_z = [None] * latent_levels
        post_c = [None] * latent_levels

        s = [None] * latent_levels

        # Generate post_z
        for i in range(latent_levels):
            net = layers.conv2D(z_list[i], 'z%d_post_1' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)
            net = layers.conv2D(net, 'z%d_post_2' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)
            net = increase_resolution(net, resolution_levels - latent_levels, name='preups_%d' % i)

            post_z[i] = net

        # Upstream path
        post_c[latent_levels - 1] = post_z[latent_levels - 1]

        for i in reversed(range(latent_levels - 1)):

            concat = post_z[i]

            net = layers.conv2D(concat, 'post_c_%d_1' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)
            net = layers.conv2D(net, 'post_c_%d_2' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)

            post_c[i] = net

        # Outputs
        for i in range(latent_levels):
            s_in = layers.conv2D(post_c[i], 'y_lvl%d' % i, num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)
            s[i] = tf.image.resize_images(s_in, image_size[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return s


def segvae_const_latent(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    """
    This is a U-NET like arch with skips before and after latent space and a rather simple decoder
    """

    n0 = kwargs.get('n0', 32)
    max_channel_power = kwargs.get('max_channel_power', 4)
    max_channels = n0*2**max_channel_power

    def increase_resolution(x, times, name):

        with tf.variable_scope(name):
            nett = x

            for i in range(times):

                nett = layers.bilinear_upsample2D(nett, 'ups_%d' % i, 2)
                nC = nett.get_shape().as_list()[3]
                nett = layers.conv2D(nett, 'z%d_post' % i, num_filters=nC, normalisation=norm, training=training)

        return nett

    n_channels = image_size[2]

    resolution_levels = kwargs.get('resolution_levels', 3)
    n0 = kwargs.get('n0', 32)

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        z_list_c = []
        pre_out = [None] * resolution_levels
        s = [None] * resolution_levels

        for i in range(resolution_levels):

            z_list_c.append(layers.conv2D(z_list[i], 'z%d_post' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training))

        pre_out[resolution_levels-1] = z_list_c[resolution_levels-1]

        for i in reversed(range(resolution_levels-1)):

            top = increase_resolution(z_list_c[i], resolution_levels-i-1, 'upsample_top_%d' % i)
            bottom = increase_resolution(pre_out[i+1], 1, 'upsample_bottom_%d' % i)
            net = tf.concat([top, bottom], axis=3)
            pre_out[i] = layers.conv2D(net, 'preout_%d' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)

        for i in range(resolution_levels):
            s_in = layers.conv2D(pre_out[i], 'y_lvl%d' % i, num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)
            s[i] = tf.image.resize_images(s_in, image_size[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return s


def prob_unet2D(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):

    x = kwargs.get('x')

    z = z_list[0]

    resolution_levels = kwargs.get('resolution_levels', 7)
    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2*n0, 4*n0,6*n0, 6*n0, 6*n0, 6*n0]

    conv_unit = layers.conv2D
    deconv_unit = lambda inp: layers.bilinear_upsample2D(inp, 'upsample', 2)

    bs = tf.shape(x)[0]
    zdim = z.get_shape().as_list()[-1]

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        enc = []

        with tf.variable_scope('encoder'):

            for ii in range(resolution_levels):

                enc.append([])

                # In first layer set input to x rather than max pooling
                if ii == 0:
                    enc[ii].append(x)
                else:
                    enc[ii].append(layers.averagepool2D(enc[ii-1][-1]))

                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_1' % ii, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_2' % ii, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_3' % ii, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels-1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj-1][-1]


                dec[jj].append(deconv_unit(next_inp))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii-1][-1]], axis=3))

                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_1' % jj, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))  # projection True to make it work with res units.
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_2' % jj, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_3' % jj, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))

        print('### z shape')
        print(z.get_shape().as_list())
        z_t = tf.reshape(z, tf.stack((bs, 1, 1, zdim)))
        print('### zt shape')
        print(z_t.get_shape().as_list())

        broadcast_z = tf.tile(z_t, (1, image_size[0], image_size[1], 1))

        print('### broadcase z shape')
        print(broadcast_z.get_shape().as_list())

        net = tf.concat([dec[-1][-1], broadcast_z], axis=-1)

        recomb = conv_unit(net, 'recomb_0', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_1', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_2', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)

        s = [layers.conv2D(recomb, 'prediction', num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)]

        return s


def det_unet2D(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):

    x = kwargs.get('x')

    z = z_list[0]

    resolution_levels = kwargs.get('resolution_levels', 7)
    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2*n0, 4*n0,6*n0, 6*n0, 6*n0, 6*n0]

    conv_unit = layers.conv2D
    deconv_unit = lambda inp: layers.bilinear_upsample2D(inp, 'upsample', 2)

    bs = tf.shape(x)[0]
    zdim = z.get_shape().as_list()[-1]

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        enc = []

        with tf.variable_scope('encoder'):

            for ii in range(resolution_levels):

                enc.append([])

                # In first layer set input to x rather than max pooling
                if ii == 0:
                    enc[ii].append(x)
                else:
                    enc[ii].append(layers.averagepool2D(enc[ii-1][-1]))

                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_1' % ii, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_2' % ii, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_3' % ii, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels-1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj-1][-1]


                dec[jj].append(deconv_unit(next_inp))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii-1][-1]], axis=3))

                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_1' % jj, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))  # projection True to make it work with res units.
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_2' % jj, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_3' % jj, num_filters=num_channels[ii], training=training, normalisation=norm, add_bias=add_bias))

        net = dec[-1][-1]

        recomb = conv_unit(net, 'recomb_0', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_1', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_2', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)

        s = [layers.conv2D(recomb, 'prediction', num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)]

        return s



def hybrid(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    """
    This is a U-NET like arch with skips before and after latent space and a rather simple decoder
    """

    use_logistic_transform = kwargs.get('use_logistic_transform')

    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2 * n0, 4 * n0, 6 * n0, 6 * n0, 6 * n0, 6 * n0]

    def increase_resolution(x, times, num_filters, name):

        with tf.variable_scope(name):
            nett = x

            for i in range(times):
                nett = layers.bilinear_upsample2D(nett, 'ups_%d' % i, 2)
                nett = layers.conv2D(nett, 'z%d_post' % i, num_filters=num_filters, normalisation=norm, training=training)

        return nett

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        resolution_levels = kwargs.get('resolution_levels', 7)
        latent_levels = kwargs.get('latent_levels', 5)
        lvl_diff = resolution_levels - latent_levels

        post_z = [None] * latent_levels
        post_c = [None] * latent_levels

        s = [None] * latent_levels

        # Generate post_z
        for i in range(latent_levels):
            net = layers.conv2D(z_list[i], 'z%d_post_1' % i, num_filters=num_channels[i], normalisation=norm, training=training)
            net = layers.conv2D(net, 'z%d_post_2' % i, num_filters=num_channels[i], normalisation=norm, training=training)
            net = increase_resolution(net, resolution_levels - latent_levels, num_filters=num_channels[i], name='preups_%d' % i)

            post_z[i] = net

        # Upstream path
        post_c[latent_levels - 1] = post_z[latent_levels - 1]

        for i in reversed(range(latent_levels - 1)):
            ups_below = layers.bilinear_upsample2D(post_c[i + 1], name='post_z%d_ups' % (i + 1), factor=2)
            ups_below = layers.conv2D(ups_below, 'post_z%d_ups_c' % (i + 1), num_filters=num_channels[i], normalisation=norm, training=training)

            concat = tf.concat([post_z[i], ups_below], axis=3, name='concat_%d' % i)

            net = layers.conv2D(concat, 'post_c_%d_1' % i, num_filters=num_channels[i+lvl_diff], normalisation=norm, training=training)
            net = layers.conv2D(net, 'post_c_%d_2' % i, num_filters=num_channels[i+lvl_diff], normalisation=norm, training=training)

            post_c[i] = net

        # Outputs
        for i in range(latent_levels):

            if use_logistic_transform:
                s_in = layers.conv2D(post_c[i], 'y_lvl%d' % i, num_filters=n_classes-1, kernel_size=(1, 1), activation=tf.identity)
                bs = tf.shape(s_in)[0]
                s_in_shape = s_in.get_shape().as_list()
                kth_channel = tf.zeros(tf.stack([bs, s_in_shape[1], s_in_shape[2], 1]))
                s_in = tf.concat([s_in, kth_channel], axis=-1)
            else:
                s_in = layers.conv2D(post_c[i], 'y_lvl%d' % i, num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)
            s[i] = tf.image.resize_images(s_in, image_size[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return s

