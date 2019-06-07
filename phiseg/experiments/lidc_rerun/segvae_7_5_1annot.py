from phiseg.model_zoo import likelihoods, posteriors, priors
import tensorflow as tf
from tfwrapper import normalisation as tfnorm

experiment_name = 'segvae_7_5_1annot'
log_dir_name = 'lidc'

# architecture
posterior = posteriors.hybrid
likelihood = likelihoods.hybrid
prior = priors.hybrid
layer_norm = tfnorm.batch_norm
use_logistic_transform = False

latent_levels = 5
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
lr_schedule_dict = {0: 1e-3} #, 80000: 1e-4}
# lr_schedule_dict = {0: 1e-4, 80000: 0.5e-4, 160000: 1e-5, 240000: 0.5e-6} #  {0: 1e-3}
deep_supervision = True
batch_size = 12
# learning_rate = 1e-3
num_iter = 5000000
annotator_range = [0]  #range(num_labels_per_subject)  # which annotators to actually use for training

# losses
KL_divergence_loss_weight = 1.0
prior_sigma_weights = [1]*resolution_levels
full_latent_dependencies = False #True  # Use only lower level z, or all levels below
exponential_weighting = True
full_covariance_list = [False, False, False, False, False]

residual_multinoulli_loss_weight = 1.0

# monitoring
do_image_summaries = True
rescale_RGB = False
validation_frequency = 500
validation_samples = 16
tensorboard_update_frequency = 100

