# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm


def prob_unet2D_arch(x,
                     training,
                     nlabels,
                     n0=32,
                     resolution_levels=7,
                     norm=tfnorm.batch_norm,
                     conv_unit=layers.conv2D,
                     deconv_unit=lambda inp: layers.bilinear_upsample2D(inp, 'upsample', 2),
                     scope_reuse=False,
                     return_net=False):


    num_channels = [n0, 2*n0, 4*n0,6*n0, 6*n0, 6*n0, 6*n0]

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


        recomb = conv_unit(dec[-1][-1], 'recomb_0', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_1', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_2', num_filters=num_channels[0], kernel_size=(1,1), training=training, normalisation=norm, add_bias=add_bias)

        s = layers.conv2D(recomb, 'prediction', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity)

        return s

