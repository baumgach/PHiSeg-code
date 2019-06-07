# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import numpy as np
import math
import glob
import os
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def flatten(tensor):
    '''
    Flatten the last N-1 dimensions of a tensor only keeping the first one, which is typically 
    equal to the number of batches. 
    Example: A tensor of shape [10, 200, 200, 32] becomes [10, 1280000] 
    '''
    rhs_dim = get_rhs_dim(tensor)
    return tf.reshape(tensor, [-1, rhs_dim])

def get_rhs_dim(tensor):
    '''
    Get the multiplied dimensions of the last N-1 dimensions of a tensor. 
    I.e. an input tensor with shape [10, 200, 200, 32] leads to an output of 1280000 
    '''
    shape = tensor.get_shape().as_list()
    return np.prod(shape[1:])


def tfndims(t):
    return len(t.get_shape().as_list())


def prepare_tensor_for_summary(img, mode, idx=0, nlabels=None, **kwargs):
    '''
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''

    if mode == 'mask':

        if img.get_shape().ndims == 3:
            V = img[idx, ...]
        elif img.get_shape().ndims == 4:
            slice = kwargs.get('slice', 10)
            V = img[idx, ..., slice]
        elif img.get_shape().ndims == 5:
            slice = kwargs.get('slice', 10)
            V = img[idx, ..., slice, 0]
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    elif mode == 'image':

        if img.get_shape().ndims == 3:
            V = img[idx, ...]
        elif img.get_shape().ndims == 4:
            V = img[idx, ...]
        elif img.get_shape().ndims == 5:
            slice = kwargs.get('slice', 10)
            V = img[idx, ..., slice, 0]
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else:
        raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

    if mode=='image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8)

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]

    V = tf.reshape(V, tf.stack((1, img_w, img_h, -1)))
    return V


def put_kernels_on_grid(images, batch_size, pad=1, min_int=None, max_int=None, **kwargs):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
      images:            [batch_size, X, Y, channels] 
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''

    mode = kwargs.get('mode', 'image')
    if mode == 'mask':
        nlabels = kwargs.get('nlabels')

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    pass
                return (i, int(n / i))

    # (grid_Y, grid_X) = factorization(images.get_shape()[0].value)
    # print('grid: %d = (%d, %d)' % (images.get_shape()[0].value, grid_Y, grid_X))

    (grid_Y, grid_X) = factorization(batch_size)
    # print('grid: %d = (%d, %d)' % (batch_size, grid_Y, grid_X))

    if mode == 'image':

        if not min_int:
            x_min = tf.reduce_min(images)
        else:
            x_min = min_int

        if not max_int:
            x_max = tf.reduce_max(images)
        else:
            x_max = max_int

        # images = tf.cast(images, tf.float32)
        # images = (images - x_min) / (x_max - x_min)
        images -= x_min
        images /= x_max

    elif mode == 'mask':
        images /= (nlabels - 1)
    else:
        raise ValueError("Unknown mode: '%s'" % mode)

    images *= 254.0  # previously had issues with intensities wrapping around, will setting to 254 fix it?
    images = tf.cast(images, tf.uint8)

    # pad X and Y
    x = tf.pad(images, tf.constant([[0, 0], [pad, pad], [pad, pad],[0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = images.get_shape().as_list()[1] + 2 * pad
    X = images.get_shape().as_list()[2] + 2 * pad

    channels = images.get_shape()[3]

    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))

    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # Transpose the image again
    x = tf.transpose(x, (0, 2, 1, 3))

    return x


from tensorflow.python import pywrap_tensorflow
def print_tensornames_in_checkpoint_file(file_name):
    """ 
    """

    reader = pywrap_tensorflow.NewCheckpointReader(file_name)

    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        print(" - tensor_name: ", key)
        #print(reader.get_tensor(key))

def get_checkpoint_weights(file_name):
    """ 
    """
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    return {n: reader.get_tensor(n) for n in reader.get_variable_to_shape_map()}


def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    if len(iteration_nums) == 0:
        return False

    latest_iteration = np.max(iteration_nums)
    return os.path.join(folder, name + '-' + str(latest_iteration))



def get_weight_variable(shape, name=None, type='xavier_uniform', regularize=True, **kwargs):

    if 'init_weights' in kwargs and kwargs['init_weights'] is not None:
        type = 'pretrained'
        logging.info('Using pretrained weights for layer: %s' % name)

    initialise_from_constant = False
    if type == 'xavier_uniform':
        initial = xavier_initializer(uniform=True, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=1.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'bilinear':
        weights = _bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'pretrained':
        initial = kwargs.get('init_weights')
        initialise_from_constant = True
        logging.info('Using pretrained weights for layer: %s' % name)
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)

    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        if initialise_from_constant:
            weight = tf.get_variable(name, initializer=initial)
        else:
            weight = tf.get_variable(name, shape=shape, initializer=initial)

    if regularize:
        tf.add_to_collection('weight_variables', weight)

    return weight



def get_bias_variable(shape, name=None, init_value=0.0, **kwargs):

    if 'init_biases' in kwargs and kwargs['init_biases'] is not None:
        initial = kwargs['init_biases']
        logging.info('Using pretrained weights for layer: %s' % name)
    else:
        initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)



def _upsample_filt(size):
    '''
    Make a nets2D bilinear kernel suitable for upsampling of the given (h, w) size.
    '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def _bilinear_upsample_weights(shape):
    '''
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    '''

    if not shape[0] == shape[1]: raise ValueError('kernel is not square')
    if not shape[2] == shape[3]: raise ValueError('input and output featuremaps must have the same size')

    kernel_size = shape[0]
    num_feature_maps = shape[2]

    weights = np.zeros(shape, dtype=np.float32)
    upsample_kernel = _upsample_filt(kernel_size)

    for i in range(num_feature_maps):
        weights[:, :, i, i] = upsample_kernel

    return weights