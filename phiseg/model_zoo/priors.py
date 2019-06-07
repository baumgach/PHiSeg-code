import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def betaVAE_bn(z_list, x, zdim_0, n_classes, generation_mode, training, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):


    resolution_levels = kwargs.get('resolution_levels', 5)
    image_size = x.get_shape().as_list()[1:3]
    final_kernel_size = [s//(2**(resolution_levels-1)) for s in image_size]

    with tf.variable_scope('prior') as scope:

        if scope_reuse:
            scope.reuse_variables()

        n0 = kwargs.get('n0', 32)

        mu_z = []
        sigma_z = []
        z = []
        # Generate pre_z's

        net = x

        for ii in range(resolution_levels-1):
            net = layers.conv2D(net, 'p_z_%d' % ii, num_filters=n0*(ii//2+1), kernel_size=(4,4), strides=(2,2), normalisation=norm, training=training)

        net = layers.conv2D(net, 'p_z_%d' % resolution_levels, num_filters=n0*8, kernel_size=final_kernel_size, strides=(1,1), padding='VALID', normalisation=norm, training=training)

        mu_z.append(layers.dense_layer(net, 'z_mu', hidden_units=zdim_0, activation=tf.identity))
        sigma_z.append(layers.dense_layer(net, 'z_sigma', hidden_units=zdim_0, activation=tf.nn.softplus))

        z.append(mu_z[0] + sigma_z[0] * tf.random_normal(tf.shape(mu_z[0]), 0, 1, dtype=tf.float32))

    return z, mu_z, sigma_z


def unet_T_L(z_list, x, zdim_0, n_classes, generation_mode, training, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):


    with tf.variable_scope('prior') as scope:

        if scope_reuse:
            scope.reuse_variables()

        full_cov_list = kwargs.get('full_cov_list', None)

        n0 = kwargs.get('n0', 32)
        max_channel_power = kwargs.get('max_channel_power', 4)
        max_channels = n0 * 2 ** max_channel_power
        latent_levels = kwargs.get('latent_levels', 4)
        resolution_levels = kwargs.get('resolution_levels', 6)

        spatial_xdim = x.get_shape().as_list()[1:3]

        full_latent_dependencies = kwargs.get('full_latent_dependencies', False)

        pre_z = [None] * resolution_levels

        mu = [None] * latent_levels
        sigma = [None] * latent_levels
        z = [None] * latent_levels

        z_ups_mat = []
        for i in range(latent_levels): z_ups_mat.append(
            [None] * latent_levels)  # encoding [original resolution][upsampled to]

        # Generate pre_z's
        for i in range(resolution_levels):

            if i == 0:
                net = x
            else:
                net = layers.reshape_pool2D_layer(pre_z[i - 1])

            net = layers.conv2D(net, 'z%d_pre_1' % i, num_filters=n0 * (i // 2 + 1), normalisation=norm, training=training)
            net = layers.conv2D(net, 'z%d_pre_2' % i, num_filters=n0 * (i // 2 + 1), normalisation=norm, training=training)

            pre_z[i] = net

        # Generate z's
        for i in reversed(range(latent_levels)):

            spatial_zdim = [d // 2 ** (i + resolution_levels - latent_levels) for d in spatial_xdim]
            spatial_cov_dim = spatial_zdim[0] * spatial_zdim[1]

            if i == latent_levels - 1:

                mu[i] = layers.conv2D(pre_z[i + resolution_levels - latent_levels], 'z%d_mu' % i, num_filters=zdim_0,
                                            activation=tf.identity)

                if full_cov_list[i] == True:

                    l = layers.dense_layer(pre_z[i + resolution_levels - latent_levels], 'z%d_sigma' % i,
                                           hidden_units=zdim_0 * spatial_cov_dim * (spatial_cov_dim + 1) // 2,
                                           activation=tf.identity)
                    l = tf.reshape(l, [-1, zdim_0, spatial_cov_dim * (spatial_cov_dim + 1) // 2])
                    Lp = tf.contrib.distributions.fill_triangular(l)
                    L = tf.linalg.set_diag(Lp, tf.nn.softplus(
                        tf.linalg.diag_part(Lp)))  # Cholesky factors must have positive diagonal

                    sigma[i] = L

                    eps = tf.random_normal(tf.shape(mu[i]))
                    eps = tf.transpose(eps, perm=[0, 3, 1, 2])
                    bs = tf.shape(x)[0]
                    eps = tf.reshape(eps, tf.stack([bs, zdim_0, -1, 1]))

                    eps_tmp = tf.matmul(sigma[i], eps)
                    eps_tmp = tf.transpose(eps_tmp, perm=[0, 2, 3, 1])
                    eps_tmp = tf.reshape(eps_tmp, [bs, spatial_zdim[0], spatial_zdim[1], zdim_0])

                    z[i] = mu[i] + eps_tmp

                else:

                    sigma[i] = layers.conv2D(pre_z[i + resolution_levels - latent_levels], 'z%d_sigma' % i,
                                                   num_filters=zdim_0, activation=tf.nn.softplus, kernel_size=(1, 1))
                    z[i] = mu[i] + sigma[i] * tf.random_normal(tf.shape(mu[i]), 0, 1, dtype=tf.float32)

            else:

                for j in reversed(range(0, i + 1)):
                    z_below_ups = layers.nearest_neighbour_upsample2D(z_ups_mat[j + 1][i + 1], factor=2)
                    z_below_ups = layers.conv2D(z_below_ups, name='z%d_ups_to_%d_c_1' % ((i + 1), (j + 1)),
                                                         num_filters=zdim_0 * n0, normalisation=norm, training=training)
                    z_below_ups = layers.conv2D(z_below_ups, name='z%d_ups_to_%d_c_2' % ((i + 1), (j + 1)),
                                                         num_filters=zdim_0 * n0, normalisation=norm, training=training)

                    z_ups_mat[j][i + 1] = z_below_ups

                if full_latent_dependencies:
                    z_input = tf.concat([pre_z[i + resolution_levels - latent_levels]] + z_ups_mat[i][(i + 1):latent_levels],
                                        axis=3, name='concat_%d' % i)
                else:
                    z_input = tf.concat([pre_z[i + resolution_levels - latent_levels], z_ups_mat[i][i + 1]], axis=3,
                                        name='concat_%d' % i)

                z_input = layers.conv2D(z_input, 'z%d_input_1' % i, num_filters=n0 * (i // 2 + 1), normalisation=norm, training=training)
                z_input = layers.conv2D(z_input, 'z%d_input_2' % i, num_filters=n0 * (i // 2 + 1), normalisation=norm, training=training)

                mu[i] = layers.conv2D(z_input, 'z%d_mu' % i, num_filters=zdim_0, activation=tf.identity,
                                            kernel_size=(1, 1))

                if full_cov_list[i] == True:
                    l = layers.dense_layer(z_input, 'z%d_sigma' % i,
                                           hidden_units=zdim_0 * spatial_cov_dim * (spatial_cov_dim + 1) // 2,
                                           activation=tf.identity)
                    l = tf.reshape(l, [-1, zdim_0, spatial_cov_dim * (spatial_cov_dim + 1) // 2])
                    Lp = tf.contrib.distributions.fill_triangular(l)
                    L = tf.linalg.set_diag(Lp, tf.nn.softplus(tf.linalg.diag_part(Lp)))

                    sigma[i] = L

                    eps = tf.random_normal(tf.shape(mu[i]))
                    eps = tf.transpose(eps, perm=[0, 3, 1, 2])
                    bs = tf.shape(x)[0]
                    eps = tf.reshape(eps, tf.stack([bs, zdim_0, -1, 1]))

                    eps_tmp = tf.matmul(sigma[i], eps)
                    eps_tmp = tf.transpose(eps_tmp, perm=[0, 2, 3, 1])
                    eps_tmp = tf.reshape(eps_tmp, [bs, spatial_zdim[0], spatial_zdim[1], zdim_0])

                    z[i] = mu[i] + eps_tmp

                else:

                    sigma[i] = layers.conv2D(z_input, 'z%d_sigma' % i, num_filters=zdim_0,
                                                   activation=tf.nn.softplus, kernel_size=(1, 1))
                    z[i] = mu[i] + sigma[i] * tf.random_normal(tf.shape(mu[i]), 0, 1, dtype=tf.float32)

            # While training use posterior samples, when generating use prior samples here
            if generation_mode:
                z_ups_mat[i][i] = z[i]
            else:
                z_ups_mat[i][i] = z_list[i]

    return z, mu, sigma


def segvae_const_latent(z_list, x, zdim_0, n_classes, generation_mode, training, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):

    n0 = kwargs.get('n0', 32)
    max_channel_power = kwargs.get('max_channel_power', 4)
    max_channels = n0*2**max_channel_power
    full_cov_list = kwargs.get('full_cov_list', None)

    resolution_levels = kwargs.get('resolution_levels', 6)

    def reduce_resolution(x, times, name):

        with tf.variable_scope(name):

            nett = x

            for ii in range(times):

                nett = layers.reshape_pool2D_layer(nett)
                nC = nett.get_shape().as_list()[3]
                nett = layers.conv2D(nett, 'down_%d' % ii, num_filters=min(nC//4, max_channels), normalisation=norm, training=training)

        return nett

    with tf.variable_scope('prior') as scope:

        spatial_xdim = x.get_shape().as_list()[1:3]
        spatial_zdim = [d // 2 ** (resolution_levels-1) for d in spatial_xdim]
        spatial_cov_dim = spatial_zdim[0]*spatial_zdim[1]

        if scope_reuse:
            scope.reuse_variables()

        n0 = kwargs.get('n0', 32)
        levels = resolution_levels

        full_latent_dependencies = kwargs.get('full_latent_dependencies', False)

        pre_z = [None] * levels
        mu = [None] * levels
        sigma = [None] * levels
        z = [None] * levels

        z_mat = []
        for i in range(levels): z_mat.append([None] * levels)  # encoding [original resolution][upsampled to]

        # Generate pre_z's
        for i in range(levels):

            if i == 0:
                net = x
            else:
                net = layers.maxpool2D(pre_z[i - 1])

            net = layers.conv2D(net, 'z%d_pre_1' % i, num_filters=n0 * (i//2+1), normalisation=norm, training=training)
            pre_z[i] = net

        # Generate z's
        for i in reversed(range(levels)):

            z_input = reduce_resolution(pre_z[i], levels - i - 1, name='reduction_%d' % i)

            if i == levels - 1:

                mu[i] = layers.conv2D(z_input, 'z%d_mu' % i, num_filters=zdim_0, activation=tf.identity)

                if full_cov_list[i] == True:

                    l = layers.dense_layer(z_input, 'z%d_sigma' % i, hidden_units=zdim_0 * spatial_cov_dim * (spatial_cov_dim + 1) // 2, activation=tf.identity)
                    l = tf.reshape(l, [-1, zdim_0, spatial_cov_dim * (spatial_cov_dim + 1) // 2])
                    Lp = tf.contrib.distributions.fill_triangular(l)
                    L = tf.linalg.set_diag(Lp, tf.nn.softplus(tf.linalg.diag_part(Lp)))  # Cholesky factors must have positive diagonal

                    sigma[i] = L

                    eps = tf.random_normal(tf.shape(mu[i]))
                    eps = tf.transpose(eps, perm=[0, 3, 1, 2])
                    bs = tf.shape(x)[0]
                    eps = tf.reshape(eps, tf.stack([bs, zdim_0, -1, 1]))

                    eps_tmp = tf.matmul(sigma[i], eps)
                    eps_tmp = tf.transpose(eps_tmp, perm=[0, 2, 3, 1])
                    eps_tmp = tf.reshape(eps_tmp, [bs, spatial_zdim[0], spatial_zdim[1], zdim_0])

                    z[i] = mu[i] + eps_tmp

                else:

                    sigma[i] = layers.conv2D(z_input, 'z%d_sigma' % i, num_filters=zdim_0, activation=tf.nn.softplus, kernel_size=(1, 1))
                    z[i] = mu[i] + sigma[i] * tf.random_normal(tf.shape(mu[i]), 0, 1, dtype=tf.float32)

            else:

                for j in reversed(range(0, i + 1)):
                    z_connect = layers.conv2D(z_mat[j + 1][i + 1], name='double_res_%d_to_%d' % ((i + 1), (j)), num_filters=2*zdim_0, normalisation=norm, training=training)
                    z_mat[j][i + 1] = z_connect

                if full_latent_dependencies:
                    z_input = tf.concat([z_input] + z_mat[i][(i + 1):levels], axis=3, name='concat_%d' % i)
                else:
                    z_input = tf.concat([z_input, z_mat[i][(i + 1)]], axis=3, name='concat_%d' % i)

                mu[i] = layers.conv2D(z_input, 'z%d_mu' % i, num_filters=zdim_0, activation=tf.identity)

                if full_cov_list[i] == True:

                    l = layers.dense_layer(z_input, 'z%d_sigma' % i, hidden_units=zdim_0 * spatial_cov_dim * (spatial_cov_dim + 1) // 2, activation=tf.identity)
                    l = tf.reshape(l, [-1, zdim_0, spatial_cov_dim * (spatial_cov_dim + 1) // 2])
                    Lp = tf.contrib.distributions.fill_triangular(l)
                    L = tf.linalg.set_diag(Lp, tf.nn.softplus(tf.linalg.diag_part(Lp)))

                    sigma[i] = L

                    eps = tf.random_normal(tf.shape(mu[i]))
                    eps = tf.transpose(eps, perm=[0, 3, 1, 2])
                    bs = tf.shape(x)[0]
                    eps = tf.reshape(eps, tf.stack([bs, zdim_0, -1, 1]))

                    eps_tmp = tf.matmul(sigma[i], eps)
                    eps_tmp = tf.transpose(eps_tmp, perm=[0, 2, 3, 1])
                    eps_tmp = tf.reshape(eps_tmp, [bs, spatial_zdim[0], spatial_zdim[1], zdim_0])

                    z[i] = mu[i] + eps_tmp

                else:

                    sigma[i] = layers.conv2D(z_input, 'z%d_sigma' % i, num_filters=zdim_0, activation=tf.nn.softplus)
                    z[i] = mu[i] + sigma[i] * tf.random_normal(tf.shape(mu[i]), 0, 1, dtype=tf.float32)

            # While training use posterior samples, when generating use prior samples here
            if generation_mode:
                z_mat[i][i] = z[i]
            else:
                z_mat[i][i] = z_list[i]

    return z, mu, sigma


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


def hybrid(z_list, x, zdim_0, n_classes, generation_mode, training, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):

    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2*n0, 4*n0,6*n0, 6*n0, 6*n0, 6*n0]

    with tf.variable_scope('prior') as scope:

        if scope_reuse:
            scope.reuse_variables()

        full_cov_list = kwargs.get('full_cov_list', None)

        n0 = kwargs.get('n0', 32)
        latent_levels = kwargs.get('latent_levels', 5)
        resolution_levels = kwargs.get('resolution_levels', 7)

        spatial_xdim = x.get_shape().as_list()[1:3]

        full_latent_dependencies = kwargs.get('full_latent_dependencies', False)

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
            spatial_cov_dim = spatial_zdim[0] * spatial_zdim[1]

            if i == latent_levels - 1:

                mu[i] = layers.conv2D(pre_z[i+resolution_levels-latent_levels], 'z%d_mu' % i, num_filters=zdim_0, activation=tf.identity)

                if full_cov_list[i] == True:

                    l = layers.dense_layer(pre_z[i+resolution_levels-latent_levels], 'z%d_sigma' % i, hidden_units=zdim_0 * spatial_cov_dim * (spatial_cov_dim + 1) // 2, activation=tf.identity)
                    l = tf.reshape(l, [-1, zdim_0, spatial_cov_dim * (spatial_cov_dim + 1) // 2])
                    Lp = tf.contrib.distributions.fill_triangular(l)
                    L = tf.linalg.set_diag(Lp, tf.nn.softplus(tf.linalg.diag_part(Lp)))  # Cholesky factors must have positive diagonal

                    sigma[i] = L

                    eps = tf.random_normal(tf.shape(mu[i]))
                    eps = tf.transpose(eps, perm=[0, 3, 1, 2])
                    bs = tf.shape(x)[0]
                    eps = tf.reshape(eps, tf.stack([bs, zdim_0, -1, 1]))

                    eps_tmp = tf.matmul(sigma[i], eps)
                    eps_tmp = tf.transpose(eps_tmp, perm=[0, 2, 3, 1])
                    eps_tmp = tf.reshape(eps_tmp, [bs, spatial_zdim[0], spatial_zdim[1], zdim_0])

                    z[i] = mu[i] + eps_tmp

                else:

                    sigma[i] = layers.conv2D(pre_z[i+resolution_levels-latent_levels], 'z%d_sigma' % i, num_filters=zdim_0, activation=tf.nn.softplus, kernel_size=(1, 1))
                    z[i] = mu[i] + sigma[i] * tf.random_normal(tf.shape(mu[i]), 0, 1, dtype=tf.float32)

            else:

                for j in reversed(range(0, i+1)):

                    z_below_ups = layers.bilinear_upsample2D(z_ups_mat[j+1][i+1], factor=2, name='ups')
                    z_below_ups = layers.conv2D(z_below_ups, name='z%d_ups_to_%d_c_1' % ((i+1), (j+1)), num_filters=zdim_0*n0, normalisation=norm, training=training)
                    z_below_ups = layers.conv2D(z_below_ups, name='z%d_ups_to_%d_c_2' % ((i+1), (j+1)), num_filters=zdim_0*n0, normalisation=norm, training=training)

                    z_ups_mat[j][i + 1] = z_below_ups

                if full_latent_dependencies:
                    z_input = tf.concat([pre_z[i+resolution_levels-latent_levels]] + z_ups_mat[i][(i+1):latent_levels], axis=3, name='concat_%d' % i)
                else:
                    z_input = tf.concat([pre_z[i+resolution_levels-latent_levels], z_ups_mat[i][i+1]], axis=3, name='concat_%d' % i)

                z_input = layers.conv2D(z_input, 'z%d_input_1' % i, num_filters=num_channels[i], normalisation=norm, training=training)
                z_input = layers.conv2D(z_input, 'z%d_input_2' % i, num_filters=num_channels[i], normalisation=norm, training=training)

                mu[i] = layers.conv2D(z_input, 'z%d_mu' % i, num_filters=zdim_0, activation=tf.identity, kernel_size=(1,1))

                if full_cov_list[i] == True:

                    l = layers.dense_layer(z_input, 'z%d_sigma' % i,
                                           hidden_units=zdim_0 * spatial_cov_dim * (spatial_cov_dim + 1) // 2,
                                           activation=tf.identity)
                    l = tf.reshape(l, [-1, zdim_0, spatial_cov_dim * (spatial_cov_dim + 1) // 2])
                    Lp = tf.contrib.distributions.fill_triangular(l)
                    L = tf.linalg.set_diag(Lp, tf.nn.softplus(tf.linalg.diag_part(Lp)))

                    sigma[i] = L

                    eps = tf.random_normal(tf.shape(mu[i]))
                    eps = tf.transpose(eps, perm=[0, 3, 1, 2])
                    bs = tf.shape(x)[0]
                    eps = tf.reshape(eps, tf.stack([bs, zdim_0, -1, 1]))

                    eps_tmp = tf.matmul(sigma[i], eps)
                    eps_tmp = tf.transpose(eps_tmp, perm=[0, 2, 3, 1])
                    eps_tmp = tf.reshape(eps_tmp, [bs, spatial_zdim[0], spatial_zdim[1], zdim_0])

                    z[i] = mu[i] + eps_tmp

                else:

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