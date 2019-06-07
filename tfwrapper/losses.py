# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
import numpy as np


def get_dice(logits, labels, epsilon=1e-10, sum_over_labels=False, sum_over_batches=False, use_hard_pred=True):
    '''
    Dice coefficient per subject per label
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :param sum_over_labels: Calculate IOU over all labels rather than for each label separately
    :param sum_over_batches: Calculate intersection and union over whole batch rather than single images
    :param use_hard_pred: If True calculates proper Dice, if False computes Dice based on softmax outputs directly which is differentiable.
    :return: tensor shaped (tf.shape(logits)[0], tf.shape(logits)[-1]) (except when sum_over_batches is on)
    '''

    ndims = logits.get_shape().ndims

    prediction = tf.nn.softmax(logits)
    if use_hard_pred:
        # This casts the predictions to binary 0 or 1
        prediction = tf.one_hot(tf.argmax(prediction, axis=-1), depth=tf.shape(prediction)[-1])

    intersection = tf.multiply(prediction, labels)

    if ndims == 5:
        reduction_axes = [1,2,3]
    else:
        reduction_axes = [1,2]

    if sum_over_batches:
        reduction_axes = [0] + reduction_axes

    if sum_over_labels:
        reduction_axes += [reduction_axes[-1] + 1]  # also sum over the last axis

    # Reduce the maps over all dimensions except the batch and the label index
    i = tf.reduce_sum(intersection, axis=reduction_axes)
    l = tf.reduce_sum(prediction, axis=reduction_axes)
    r = tf.reduce_sum(labels, axis=reduction_axes)

    dice_per_img_per_lab = 2 * i / (l + r + epsilon)

    return dice_per_img_per_lab


def dice_loss(logits, labels, epsilon=1e-10, **kwargs):
    '''
    The dice loss is always 1 - dice, however, there are many ways to calculate the dice. Basically, there are 
    three sums involved: 1) over the pixels, 2) over the labels, 3) over the images in a batch. These sums 
    can be arranged differently to obtain different behaviour. The behaviour can be controlled either by providing
    the 'mode' variable, or by setting the parameters directly. 
    
    Selecting the parameters directly:
    :param per_structure: <True|False> If True the Dice is calculated for each label separately first and then averaged.
    :param sum_over_batches: <True|False> If True the Dice is calculated for each batch separately then averaged.

    Selecting the mode:
    :param mode: <'macro'|'macro_robust'|'micro'>
                     macro: Calculate Dice for each label separately then average. This may cause problems
                            if a structure is completely missing from the image. Even if correctly predicted
                            the dice will evaluate to 0/epsilon = 0. However, this method automatically tackles
                            class imbalance, because each structure contributes equally to the final Dice. 
                     macro_robust: The above calculation can be made more robust by summing over all images in a 
                                   minibatch. If the label appear at least in one image in the batch and is perfectly
                                   predicted, the Dice will evaluate to 1 as expected. 
                     micro: Calculate Dice for all labels together. This doesn't have the problems of macro for missing
                            labels. However, it is sensitive to class imbalance because each label contributes 
                            by how often it appears in the data. 
                            
    The above are equivalent to F1 score in macro/micro mode (see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
                            
    Other parameters:
    :param logits: Network output
    :param labels: Ground-truth labels
    :param only_foreground: <True|False> Sometimes it can be beneficial to ignore label 0 for the optimisation
    :epsilon: <float> To avoid division by zero in Dice calculation. 

    '''

    only_foreground = kwargs.get('only_foreground', False)
    mode = kwargs.get('mode', None)
    if mode == 'macro':
        sum_over_labels = False
        sum_over_batches = False
    elif mode == 'macro_robust':
        sum_over_labels = False
        sum_over_batches = True
    elif mode == 'micro':
        sum_over_labels = True
        sum_over_batches = False
    elif mode is None:
        sum_over_labels = kwargs.get('per_structure')  # Intentionally no default value
        sum_over_batches = kwargs.get('sum_over_batches', False)
    else:
        raise  ValueError("Encountered unexpected 'mode' in dice_loss: '%s'" % mode)


    with tf.name_scope('dice_loss'):

        dice_per_img_per_lab = get_dice(logits=logits,
                                        labels=labels,
                                        epsilon=epsilon,
                                        sum_over_labels=sum_over_labels,
                                        sum_over_batches=sum_over_batches,
                                        use_hard_pred=False)

        if only_foreground:
            if sum_over_batches:
                loss = 1 - tf.reduce_mean(dice_per_img_per_lab[1:])
            else:
                loss = 1 - tf.reduce_mean(dice_per_img_per_lab[:, 1:])
        else:
            loss = 1 - tf.reduce_mean(dice_per_img_per_lab)

    return loss



def cross_entropy_loss(logits, labels, use_sigmoid=False):
    '''
    Simple wrapper for the normal tensorflow cross entropy loss 
    '''

    if use_sigmoid:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    else:
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))



def pixel_wise_cross_entropy_loss_weighted(logits, labels, class_weights):
    '''
    Weighted cross entropy loss, with a weight per class
    :param logits: Network output before softmax
    :param labels: Ground truth masks
    :param class_weights: A list of the weights for each class
    :return: weighted cross entropy loss
    '''

    n_class = len(class_weights)

    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(labels, [-1, n_class])

    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    weight_map = tf.multiply(flat_labels, class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    weighted_loss = tf.multiply(loss_map, weight_map)

    loss = tf.reduce_mean(weighted_loss)

    return loss



