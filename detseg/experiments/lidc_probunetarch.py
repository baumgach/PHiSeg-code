from detseg.network_zoo import nets2D
import tensorflow as tf
import os
import config.system as sys_config
from tfwrapper import normalisation

experiment_name = 'detUNET_bettereval'
log_dir_name = 'lidc'

# Model settings
network = nets2D.prob_unet2D_arch
normalisation = normalisation.batch_norm
resolution_levels = 7

# Data settings
data_identifier = 'lidc'
preproc_folder = '/srv/glusterfs/baumgach/preproc_data/lidc'
data_root = '/itet-stor/baumgach/bmicdatasets-originals/Originals/LIDC-IDRI/data_lidc.pickle'
dimensionality_mode = '2D'
image_size = (128, 128)
nlabels = 2
num_labels_per_subject = 4

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': nlabels}

# Network settings
n0 = 32

# Cost function
weight_decay = 0.0
loss_type = 'crossentropy'  # 'dice_micro'/'dice_macro'/'dice_macro_robust'/'crossentropy'

# Training settings
annotator_range = [0]
batch_size = 12
n_accum_grads = 1
lr_schedule_dict = {0: 1e-3}
optimizer_handle = tf.train.AdamOptimizer
beta1=0.9
beta2=0.999
divide_lr_frequency = None
warmup_training = False
momentum = None

# Augmentation
do_augmentations = True

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_iterations = 1000000
train_eval_frequency = 50
val_eval_frequency = 50
