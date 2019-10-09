from phiseg.model_zoo import likelihoods, posteriors, priors
import tensorflow as tf
from tfwrapper import normalisation as tfnorm

experiment_name = 'phiseg_7_1'
log_dir_name = 'lidc'

# architecture
posterior = posteriors.phiseg
likelihood = likelihoods.phiseg
prior = priors.phiseg
layer_norm = tfnorm.batch_norm
use_logistic_transform = False

latent_levels = 1
resolution_levels = 7
n0 = 32
zdim0 = 2
max_channel_power = 4  # max number of channels will be n0*2**max_channel_power

# Data settings
data_identifier = 'lidc'
preproc_folder = '/srv/glusterfs/baumgach/preproc_data/lidc'
data_root = '/itet-stor/baumgach/bmicdatasets-originals/Originals/LIDC-IDRI/data_lidc.pickle'
dimensionality_mode = '2D'
image_size = (128, 128, 1)
nlabels = 2
num_labels_per_subject = 4

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': nlabels}

# training
optimizer = tf.train.AdamOptimizer
lr_schedule_dict = {0: 1e-3}
deep_supervision = True
batch_size = 12
num_iter = 5000000
annotator_range = range(num_labels_per_subject)  # which annotators to actually use for training

# losses
KL_divergence_loss_weight = 1.0
exponential_weighting = True

residual_multinoulli_loss_weight = 1.0

# monitoring
do_image_summaries = True
rescale_RGB = False
validation_frequency = 500
validation_samples = 16
num_validation_images = 100 #'all'
tensorboard_update_frequency = 100

