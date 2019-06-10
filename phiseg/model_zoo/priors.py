import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def prob_unet2D(z_list, x, zdim_0, n_classes, generation_mode, training, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):

    resolution_levels = kwargs.get('resolution_levels', 7)

    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2*n0, 4*n0,6*n0, 6*n0, 6*n0, 6*n0]

    conv_unit = layers.conv2D

    with tf.variable_scope('prior') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        enc = []

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

        mu_p = conv_unit(enc[-1][-1], 'pre_mu', num_filters=zdim_0, kernel_size=(1, 1), activation=tf.identity)
        mu = [layers.global_averagepool2D(mu_p)]

        sigma_p = conv_unit(enc[-1][-1], 'pre_sigma', num_filters=zdim_0, kernel_size=(1, 1), activation=tf.nn.softplus)
        sigma = [layers.global_averagepool2D(sigma_p)]

        z = [mu[0] + sigma[0] * tf.random_normal(tf.shape(mu[0]), 0, 1, dtype=tf.float32)]

        return z, mu, sigma


def phiseg(z_list, x, zdim_0, n_classes, generation_mode, training, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):

    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2*n0, 4*n0,6*n0, 6*n0, 6*n0, 6*n0]

    with tf.variable_scope('prior') as scope:

        if scope_reuse:
            scope.reuse_variables()

        n0 = kwargs.get('n0', 32)
        latent_levels = kwargs.get('latent_levels', 5)
        resolution_levels = kwargs.get('resolution_levels', 7)

        spatial_xdim = x.get_shape().as_list()[1:3]

        pre_z = [None] * resolution_levels

        mu = [None] * latent_levels
        sigma = [None] * latent_levels
        z = [None] * latent_levels

        z_ups_mat = []
        for i in range(latent_levels): z_ups_mat.append([None]*latent_levels)  # encoding [original resolution][upsampled to]

        # Generate pre_z's
        for i in range(resolution_levels):

            if i == 0:
                net = x
            else:
                net = layers.averagepool2D(pre_z[i-1])

            net = layers.conv2D(net, 'z%d_pre_1' % i, num_filters=num_channels[i], normalisation=norm, training=training)
            net = layers.conv2D(net, 'z%d_pre_2' % i, num_filters=num_channels[i], normalisation=norm, training=training)
            net = layers.conv2D(net, 'z%d_pre_3' % i, num_filters=num_channels[i], normalisation=norm, training=training)

            pre_z[i] = net

        # Generate z's
        for i in reversed(range(latent_levels)):

            spatial_zdim = [d // 2 ** (i + resolution_levels - latent_levels) for d in spatial_xdim]

            if i == latent_levels - 1:

                mu[i] = layers.conv2D(pre_z[i+resolution_levels-latent_levels], 'z%d_mu' % i, num_filters=zdim_0, activation=tf.identity)

                sigma[i] = layers.conv2D(pre_z[i+resolution_levels-latent_levels], 'z%d_sigma' % i, num_filters=zdim_0, activation=tf.nn.softplus, kernel_size=(1, 1))
                z[i] = mu[i] + sigma[i] * tf.random_normal(tf.shape(mu[i]), 0, 1, dtype=tf.float32)

            else:

                for j in reversed(range(0, i+1)):

                    z_below_ups = layers.bilinear_upsample2D(z_ups_mat[j+1][i+1], factor=2, name='ups')
                    z_below_ups = layers.conv2D(z_below_ups, name='z%d_ups_to_%d_c_1' % ((i+1), (j+1)), num_filters=zdim_0*n0, normalisation=norm, training=training)
                    z_below_ups = layers.conv2D(z_below_ups, name='z%d_ups_to_%d_c_2' % ((i+1), (j+1)), num_filters=zdim_0*n0, normalisation=norm, training=training)

                    z_ups_mat[j][i + 1] = z_below_ups

                z_input = tf.concat([pre_z[i+resolution_levels-latent_levels], z_ups_mat[i][i+1]], axis=3, name='concat_%d' % i)

                z_input = layers.conv2D(z_input, 'z%d_input_1' % i, num_filters=num_channels[i], normalisation=norm, training=training)
                z_input = layers.conv2D(z_input, 'z%d_input_2' % i, num_filters=num_channels[i], normalisation=norm, training=training)

                mu[i] = layers.conv2D(z_input, 'z%d_mu' % i, num_filters=zdim_0, activation=tf.identity, kernel_size=(1,1))

                sigma[i] = layers.conv2D(z_input, 'z%d_sigma' % i, num_filters=zdim_0, activation=tf.nn.softplus, kernel_size=(1, 1))
                z[i] = mu[i] + sigma[i] * tf.random_normal(tf.shape(mu[i]), 0, 1, dtype=tf.float32)

            # While training use posterior samples, when generating use prior samples here
            if generation_mode:
                z_ups_mat[i][i] = z[i]
            else:
                z_ups_mat[i][i] = z_list[i]

    return z, mu, sigma

def dummy(z_list, x, zdim_0, n_classes, generation_mode, training, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    return [None, None, None]