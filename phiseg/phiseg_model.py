import tensorflow as tf
from tfwrapper import utils as tfutils

import utils

import numpy as np
import os
import time
from medpy.metric import dc

from config import system as sys_config

sys_config.setup_GPU_environment()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class phiseg():

    def __init__(self, exp_config):

        self.exp_config = exp_config
        self.checks()

        # define the input place holder
        self.x_inp = tf.placeholder(tf.float32, shape=[None] + list(self.exp_config.image_size), name='x_input')
        self.s_inp = tf.placeholder(tf.uint8, shape=[None] + list(self.exp_config.image_size[0:2]), name='s_input')

        self.s_inp_oh = tf.one_hot(self.s_inp, depth=exp_config.nlabels)

        self.training_pl = tf.placeholder(tf.bool, shape=[], name='training_time')
        self.lr_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')


        # CREATE NETwORKS

        self.z_list, self.mu_list, self.sigma_list = self.exp_config.posterior(
            self.x_inp,
            self.s_inp_oh,
            exp_config.zdim0,
            training=self.training_pl,
            n0=exp_config.n0,
            resolution_levels=exp_config.resolution_levels,
            latent_levels=exp_config.latent_levels,
            norm=exp_config.layer_norm
        )

        self.prior_z_list, self.prior_mu_list, self.prior_sigma_list = self.exp_config.prior(
            self.z_list,
            self.x_inp,
            zdim_0=exp_config.zdim0,
            n_classes=self.exp_config.nlabels,
            training=self.training_pl,
            n0=exp_config.n0,
            generation_mode=False,
            resolution_levels=exp_config.resolution_levels,
            latent_levels=exp_config.latent_levels,
            norm=exp_config.layer_norm
        )

        self.prior_z_list_gen, self.prior_mu_list_gen, self.prior_sigma_list_gen = self.exp_config.prior(
            self.z_list,
            self.x_inp,
            zdim_0=exp_config.zdim0,
            n_classes=self.exp_config.nlabels,
            training=self.training_pl,
            n0=exp_config.n0,
            generation_mode=True,
            scope_reuse=True,
            resolution_levels=exp_config.resolution_levels,
            latent_levels=exp_config.latent_levels,
            norm=exp_config.layer_norm
        )

        self.s_out_list = self.exp_config.likelihood(self.z_list,
                                                     self.training_pl,
                                                     n0=exp_config.n0,
                                                     n_classes=exp_config.nlabels,
                                                     resolution_levels=exp_config.resolution_levels,
                                                     latent_levels=exp_config.latent_levels,
                                                     image_size=exp_config.image_size,
                                                     norm=exp_config.layer_norm,
                                                     x=self.x_inp)  # This is only needed for probUNET!

        self.s_out_sm_list = [None]*self.exp_config.latent_levels
        for ii in range(self.exp_config.latent_levels):
            self.s_out_sm_list[ii] = tf.nn.softmax(self.s_out_list[ii])

        self.s_out_eval_list = self.exp_config.likelihood(self.prior_z_list_gen,
                                                     self.training_pl,
                                                     scope_reuse=True,
                                                     n0=exp_config.n0,
                                                     n_classes=exp_config.nlabels,
                                                     resolution_levels=exp_config.resolution_levels,
                                                     latent_levels=exp_config.latent_levels,
                                                     image_size=exp_config.image_size,
                                                     norm=exp_config.layer_norm,
                                                     x=self.x_inp)   # This is only needed for probUNET!

        self.s_out_eval_sm_list = [None]*self.exp_config.latent_levels
        for ii in range(self.exp_config.latent_levels):
            self.s_out_eval_sm_list[ii] = tf.nn.softmax(self.s_out_eval_list[ii])


        # Create final output from output list
        self.s_out = self._aggregate_output_list(self.s_out_list, use_softmax=False)
        self.s_out_eval = self._aggregate_output_list(self.s_out_eval_list, use_softmax=False)

        self.s_out_eval_sm = tf.nn.softmax(self.s_out_eval)

        self.eval_xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.s_inp_oh, logits=self.s_out_eval)

        # Add losses
        self.loss_dict = {}
        self.loss_tot = 0

        logging.info('ADDING LOSSES')
        if hasattr(self.exp_config, 'residual_multinoulli_loss_weight') and exp_config.residual_multinoulli_loss_weight is not None:
            logging.info(' - Adding residual multinoulli loss')
            self.add_residual_multinoulli_loss()

        if hasattr(self.exp_config, 'KL_divergence_loss_weight') and exp_config.KL_divergence_loss_weight is not None:
            logging.info(' - Adding hierarchical KL loss')
            self.add_hierarchical_KL_div_loss()

        if hasattr(self.exp_config, 'weight_decay_weight') and exp_config.weight_decay_weight is not None:
            logging.info(' - Adding weight decay')
            self.add_weight_decay()

        self.loss_dict['total_loss'] = self.loss_tot

        self.global_step = tf.train.get_or_create_global_step()

        # Create Update Operation
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if exp_config.optimizer == tf.train.MomentumOptimizer:
                optimizer = exp_config.optimizer(learning_rate=self.lr_pl, momentum=0.9, use_nesterov=True)
            else:
                optimizer = exp_config.optimizer(learning_rate=self.lr_pl)
            self.train_step = optimizer.minimize(self.loss_tot, global_step=self.global_step)

        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=3)
        self.saver_best_loss = tf.train.Saver(max_to_keep=2)
        self.saver_best_dice = tf.train.Saver(max_to_keep=2)
        self.saver_best_ged = tf.train.Saver(max_to_keep=2)
        self.saver_best_ncc = tf.train.Saver(max_to_keep=2)

        # Settings to optimize GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        # Create a session for running Ops on the Graph.
        self.sess = tf.Session(config=config)


    def checks(self):

        pass
        # if hasattr(self.exp_config, 'residual_multinoulli_loss_weight') and not self.exp_config.discrete_data:
        #     raise ValueError('Invalid settings in exp_config: residual_multinoulli_loss requires discrete_data to be True.')

    def train(self, data):

        # Sort out proper logging
        self._setup_log_dir_and_continue_mode()

        # Create tensorboard summaries
        self._make_tensorboard_summaries()

        # Initialise variables
        self.sess.run(tf.global_variables_initializer())

        # Restore session if there is one
        if self.continue_run:
            self.saver.restore(self.sess, self.init_checkpoint_path)

        self.best_dice = -1
        self.best_loss = np.inf
        self.best_ged = np.inf
        self.best_ncc = -1

        for step in range(self.init_step, self.exp_config.num_iter):

            # Get current learning rate from lr_dict
            lr_key, _ = utils.find_floor_in_list(self.exp_config.lr_schedule_dict.keys(), step)
            lr = self.exp_config.lr_schedule_dict[lr_key]


            x_b, s_b = data.train.next_batch(self.exp_config.batch_size)
            _, loss_tot_eval = self.sess.run([self.train_step, self.loss_tot], feed_dict={self.x_inp: x_b,
                                                                                          self.s_inp: s_b,
                                                                                          self.training_pl: True,
                                                                                          self.lr_pl: lr})

            if step % self.exp_config.tensorboard_update_frequency == 0:

                summary_str = self.sess.run(self.summary, feed_dict={self.x_inp: x_b, self.s_inp: s_b, self.training_pl: False, self.lr_pl: lr})
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()

            if step % self.exp_config.validation_frequency == 0:

                self._do_validation(data)


    def KL_two_gauss_with_diag_cov(self, mu0, sigma0, mu1, sigma1):

        sigma0_fs = tf.square(tfutils.flatten(sigma0))
        sigma1_fs = tf.square(tfutils.flatten(sigma1))

        logsigma0_fs = tf.log(sigma0_fs + 1e-10)
        logsigma1_fs = tf.log(sigma1_fs + 1e-10)

        mu0_f = tfutils.flatten(mu0)
        mu1_f = tfutils.flatten(mu1)

        return tf.reduce_mean(
            0.5*tf.reduce_sum(tf.divide(sigma0_fs + tf.square(mu1_f - mu0_f), sigma1_fs + 1e-10)
                              + logsigma1_fs
                              - logsigma0_fs
                              - 1, axis=1)
        )


    def multinoulli_loss_with_logits(self, x_gt, y_target):

        bs = tf.shape(x_gt)[0]

        x_f = tf.reshape(x_gt, tf.stack([bs, -1, self.exp_config.nlabels]))
        y_f = tf.reshape(y_target, tf.stack([bs, -1, self.exp_config.nlabels]))

        return tf.reduce_mean(
            tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=x_f, logits=y_f), axis=1)
        )


    def add_residual_multinoulli_loss(self):

        # TODO: move s_accum outside of this function

        self.s_accum = [None] * self.exp_config.latent_levels

        for ii, s_ii in zip(reversed(range(self.exp_config.latent_levels)),
                            reversed(self.s_out_list)):

            if ii == self.exp_config.latent_levels-1:

                self.s_accum[ii] = s_ii
                self.loss_dict['residual_multinoulli_loss_lvl%d' % ii] = self.multinoulli_loss_with_logits(self.s_inp_oh, self.s_accum[ii])

            else:

                self.s_accum[ii] = self.s_accum[ii+1] + s_ii
                self.loss_dict['residual_multinoulli_loss_lvl%d' % ii] = self.multinoulli_loss_with_logits(self.s_inp_oh, self.s_accum[ii])

            logging.info(' -- Added residual multinoulli loss at level %d' % (ii))

            self.loss_tot += self.exp_config.residual_multinoulli_loss_weight * self.loss_dict['residual_multinoulli_loss_lvl%d' % ii]


    def add_hierarchical_KL_div_loss(self):

        prior_sigma_list = self.prior_sigma_list
        prior_mu_list = self.prior_mu_list

        if self.exp_config.exponential_weighting:
            level_weights = [4**i for i in list(range(self.exp_config.latent_levels))]
        else:
            level_weights = [1]*self.exp_config.latent_levels

        for ii, mu_i, sigma_i in zip(reversed(range(self.exp_config.latent_levels)),
                                     reversed(self.mu_list),
                                     reversed(self.sigma_list)):

            self.loss_dict['KL_divergence_loss_lvl%d' % ii] = level_weights[ii]*self.KL_two_gauss_with_diag_cov(
                mu_i,
                sigma_i,
                prior_mu_list[ii],
                prior_sigma_list[ii])

            logging.info(' -- Added hierarchical loss with at level %d with alpha_%d=%d' % (ii,ii, level_weights[ii]))

            self.loss_tot += self.exp_config.KL_divergence_loss_weight * self.loss_dict['KL_divergence_loss_lvl%d' % ii]


    def add_weight_decay(self):

        weights_norm = tf.reduce_sum(
            input_tensor= tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )

        self.loss_dict['weight_decay'] = self.exp_config.weight_decay_weight*weights_norm
        self.loss_tot += self.loss_dict['weight_decay']



    def _aggregate_output_list(self, output_list, use_softmax=True):

        s_accum = output_list[-1]
        for i in range(len(output_list) - 1):
            s_accum += output_list[i]
        if use_softmax:
            return tf.nn.softmax(s_accum)
        return s_accum

    def generate_samples_from_z(self, z_list, x_in, output_all_levels=False):

        feed_dict = { i: d for i, d in zip(self.z_list, z_list)}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_inp] = x_in

        if output_all_levels:
            return self.sess.run(self.s_out_list, feed_dict=feed_dict)
        else:
            return self.sess.run(self.s_out, feed_dict=feed_dict)


    def generate_prior_samples(self, x_in, return_params=False):

        z_samples = self.sess.run(self.prior_z_list_gen, feed_dict={self.training_pl: False, self.x_inp: x_in})

        if return_params:
            prior_mu_list = self.sess.run(self.prior_mu_list_gen, feed_dict={self.training_pl: False, self.x_inp: x_in})
            prior_sigma_list = self.sess.run(self.prior_sigma_list_gen, feed_dict={self.training_pl: False, self.x_inp: x_in})
            return z_samples, prior_mu_list, prior_sigma_list
        else:
            return z_samples


    def predict(self, x_in, num_samples=50, return_softmax=False):

        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_inp] = x_in
        # feed_dict[self.s_inp] = np.zeros([x_in.shape[0]] + self.s_inp.get_shape().as_list()[1:])  # dummy

        cumsum_sm = self.sess.run(self.s_out_eval_sm, feed_dict=feed_dict)

        for i in range(num_samples-1):
            # print(' - sample: %d' % (i+1))
            cumsum_sm += self.sess.run(self.s_out_eval_sm, feed_dict=feed_dict)

        if return_softmax:
            return np.argmax(cumsum_sm, axis=-1), cumsum_sm / num_samples

        return np.argmax(cumsum_sm, axis=-1)


    def predict_segmentation_sample(self, x_in, return_softmax=False):

        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_inp] = x_in

        if return_softmax:
            return self.sess.run(self.s_out_eval_sm, feed_dict=feed_dict)
        return np.argmax(self.sess.run(self.s_out_eval, feed_dict=feed_dict), axis=-1)


    def predict_segmentation_sample_levels(self, x_in, return_softmax=False):

        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_inp] = x_in

        if return_softmax:
            return self.sess.run(self.s_out_eval_sm_list, feed_dict=feed_dict)
        return self.sess.run(self.s_out_eval_list, feed_dict=feed_dict)


    def predict_segmentation_sample_variance_sm_cov(self, x_in, num_samples):

        # assert self.exp_config.use_logistic_transform is True, 'predict variance is only implemented for logistic transform nets'

        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_inp] = x_in

        segms = []
        for _ in range(num_samples):
            segms.append(self.sess.run(self.s_out_eval, feed_dict=feed_dict))

        segm_arr = np.squeeze(np.asarray(segms)) # num_samples x size_1 x size_2 x n_classes - 1
        segm_arr = segm_arr[...,:-1]
        segm_arr = segm_arr.transpose((1,2,3,0))
        segm_arr = np.clip(segm_arr, 1e-5, 1 - (1e-5))

        corr_mat = np.einsum('ghij,ghkj->ghik', segm_arr, segm_arr) / num_samples
        mu_mat = np.mean(segm_arr, axis=-1)
        outer_mu = np.einsum('ghi,ghj->ghij', mu_mat, mu_mat)
        cov_mat = corr_mat - outer_mu  # 128, 128, nlabels -1 x nlabels -1
        # det_cov_mat = np.linalg.det(cov_mat)
        det_cov_mat, _ = np.linalg.eig(cov_mat) # 128, 128, nlabels -1


        return np.sum(det_cov_mat, axis=-1)


    def predict_segmentation_sample_variance_sm_cov_bf(self, x_in, num_samples):

        # assert self.exp_config.use_logistic_transform is True, 'predict variance is only implemented for logistic transform nets'

        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_inp] = x_in

        segms = []
        for _ in range(num_samples):
            segms.append(self.sess.run(self.s_out_eval_sm, feed_dict=feed_dict))

        segm_arr = np.squeeze(np.asarray(segms))  # num_samples x size_1 x size_2 x n_classes - 1
        # segm_arr = segm_arr[..., :-1]

        segm_arr = segm_arr.transpose((1, 2, 3, 0))  # size_1, size_2, n_classes-1, num_samples

        out = np.zeros((self.exp_config.image_size[0:2]))

        for ii in range(self.exp_config.image_size[0]):
            for jj in range(self.exp_config.image_size[1]):
                cov = np.cov(segm_arr[ii, jj, :, :])
                out[ii, jj] = np.linalg.det(cov)

        return out


    def get_crossentropy_error_map(self, s_gt, x_in, num_samples=100):

        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_inp] = x_in
        feed_dict[self.s_inp] = s_gt

        err_maps = []
        for _ in range(num_samples):
            err_maps.append(self.sess.run(self.eval_xent, feed_dict=feed_dict))

        err_maps_arr = np.asarray(err_maps)

        return np.mean(err_maps_arr, axis=0)


    def predict_mean_variance_and_error_maps(self, s_gt, x_in, num_samples):

        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_inp] = x_in
        feed_dict[self.s_inp] = s_gt

        err_maps = []
        segms = []

        for _ in range(num_samples):
            seg, err = self.sess.run([self.s_out_eval_sm, self.eval_xent], feed_dict=feed_dict)

            err_maps.append(err)
            segms.append(seg)

        err_maps_arr = np.squeeze(np.asarray(err_maps))
        segm_arr = np.squeeze(np.asarray(segms))

        vars = np.std(segm_arr, axis=0)
        vars = np.mean(vars, axis=-1)

        means = np.argmax(np.mean(segm_arr, 0), axis=-1)

        errs = np.mean(err_maps_arr, axis=0)

        return means, vars, errs


    def generate_samples_from_prior(self, x_in, output_all_levels=False):

        z_samples = self.generate_prior_samples(x_in)
        return self.generate_samples_from_z(z_samples, output_all_levels)


    def generate_posterior_samples(self, x_in, s_in, return_params=False):

        z_samples = self.sess.run(self.z_list, feed_dict={self.training_pl: False,
                                                          self.x_inp: x_in,
                                                          self.s_inp: s_in})

        if return_params:
            mu_list = self.sess.run(self.mu_list, feed_dict={self.training_pl: False, self.x_inp: x_in, self.s_inp: s_in})
            sigma_list = self.sess.run(self.sigma_list, feed_dict={self.training_pl: False, self.x_inp: x_in, self.s_inp: s_in})
            return z_samples, mu_list, sigma_list
        else:
            return z_samples


    def generate_all_output_levels(self, x_in):

        y_list = self.sess.run(self.s_out_list, feed_dict={self.x_inp: x_in,
                                                           self.training_pl: False})
        return y_list


    def load_weights(self, log_dir=None, type='latest', **kwargs):

        if not log_dir:
            log_dir = self.log_dir

        if type == 'latest':
            init_checkpoint_path = tfutils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
        elif type == 'best_dice':
            init_checkpoint_path = tfutils.get_latest_model_checkpoint_path(log_dir, 'model_best_dice.ckpt')
        elif type == 'best_loss':
            init_checkpoint_path = tfutils.get_latest_model_checkpoint_path(log_dir, 'model_best_loss.ckpt')
        elif type == 'best_ged':
            init_checkpoint_path = tfutils.get_latest_model_checkpoint_path(log_dir, 'model_best_ged.ckpt')
        elif type == 'iter':
            assert 'iteration' in kwargs, "argument 'iteration' must be provided for type='iter'"
            iteration = kwargs['iteration']
            init_checkpoint_path = os.path.join(log_dir, 'model.ckpt-%d' % iteration)
        else:
            raise ValueError('Argument type=%s is unknown. type can be latest/iter.' % type)

        self.saver.restore(self.sess, init_checkpoint_path)


    ### HELPER FUNCTIONS #################

    def _do_validation(self, data):

        global_step = self.sess.run(self.global_step) - 1

        checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_file, global_step=global_step)

        val_x, val_s = data.validation.next_batch(self.exp_config.batch_size)
        val_losses_out = self.sess.run(list(self.loss_dict.values()),
                                       feed_dict={self.x_inp: val_x, self.s_inp: val_s, self.training_pl: False}
                                       )

        # Note that val_losses_out are now sorted in the same way as loss_dict,
        tot_loss_index = list(self.loss_dict.keys()).index('total_loss')
        val_loss_tot = val_losses_out[tot_loss_index]

        train_x, train_s = data.train.next_batch(self.exp_config.batch_size)
        train_losses_out = self.sess.run(list(self.loss_dict.values()),
                                         feed_dict={self.x_inp: train_x, self.s_inp: train_s, self.training_pl: False}
                                         )


        logging.info('----- Step: %d ------' % global_step)
        logging.info('BATCH VALIDATION:')
        for ii, loss_name in enumerate(self.loss_dict.keys()):
            logging.info('%s | training: %f | validation: %f' % (loss_name, train_losses_out[ii], val_losses_out[ii]))

        # Evaluate validation Dice:

        start_dice_val = time.time()
        num_batches = 0

        dice_list = []
        elbo_list = []
        ged_list = []
        ncc_list = []

        N = data.validation.images.shape[0] if self.exp_config.num_validation_images == 'all' else self.exp_config.num_validation_images

        for ii in range(N):

            # logging.info(ii)

            x = data.validation.images[ii, ...].reshape([1] + list(self.exp_config.image_size))
            s_gt_arr = data.validation.labels[ii, ...]
            s = s_gt_arr[:,:,np.random.choice(self.exp_config.annotator_range)]

            x_b = np.tile(x, [self.exp_config.validation_samples, 1, 1, 1])
            s_b = np.tile(s, [self.exp_config.validation_samples, 1, 1])

            feed_dict = {}
            feed_dict[self.training_pl] = False
            feed_dict[self.x_inp] = x_b
            feed_dict[self.s_inp] = s_b

            s_pred_sm_arr, elbo = self.sess.run([self.s_out_eval_sm, self.loss_tot], feed_dict=feed_dict)

            s_pred_sm_mean_ = np.mean(s_pred_sm_arr, axis=0)

            s_pred_arr = np.argmax(s_pred_sm_arr, axis=-1)
            s_gt_arr_r = s_gt_arr.transpose((2, 0, 1))  # num gts x X x Y

            s_gt_arr_r_sm = utils.convert_batch_to_onehot(s_gt_arr_r, self.exp_config.nlabels)  # num gts x X x Y x nlabels

            ged = utils.generalised_energy_distance(s_pred_arr, s_gt_arr_r,
                                                    nlabels=self.exp_config.nlabels-1,
                                                    label_range=range(1, self.exp_config.nlabels))

            ncc = utils.variance_ncc_dist(s_pred_sm_arr, s_gt_arr_r_sm)

            s_ = np.argmax(s_pred_sm_mean_, axis=-1)

            # Write losses to list
            per_lbl_dice = []
            for lbl in range(self.exp_config.nlabels):
                binary_pred = (s_ == lbl) * 1
                binary_gt = (s == lbl) * 1

                if np.sum(binary_gt) == 0 and np.sum(binary_pred) == 0:
                    per_lbl_dice.append(1)
                elif np.sum(binary_pred) > 0 and np.sum(binary_gt) == 0 or np.sum(binary_pred) == 0 and np.sum(binary_gt) > 0:
                    per_lbl_dice.append(0)
                else:
                    per_lbl_dice.append(dc(binary_pred, binary_gt))

            num_batches += 1

            dice_list.append(per_lbl_dice)
            elbo_list.append(elbo)
            ged_list.append(ged)
            ncc_list.append(ncc)

        dice_arr = np.asarray(dice_list)
        per_structure_dice = dice_arr.mean(axis=0)

        avg_dice = np.mean(dice_arr)
        avg_elbo = utils.list_mean(elbo_list)
        avg_ged = utils.list_mean(ged_list)
        avg_ncc = utils.list_mean(ncc_list)

        logging.info('FULL VALIDATION (%d images):' % N)
        logging.info(' - Mean foreground dice: %.4f' % np.mean(per_structure_dice))
        logging.info(' - Mean (neg.) ELBO: %.4f' % avg_elbo)
        logging.info(' - Mean GED: %.4f' % avg_ged)
        logging.info(' - Mean NCC: %.4f' % avg_ncc)

        logging.info('@ Running through validation set took: %.2f secs' % (time.time() - start_dice_val))

        if np.mean(per_structure_dice) >= self.best_dice:
            self.best_dice = np.mean(per_structure_dice)
            logging.info('New best validation Dice! (%.3f)' % self.best_dice)
            best_file = os.path.join(self.log_dir, 'model_best_dice.ckpt')
            self.saver_best_dice.save(self.sess, best_file, global_step=global_step)

        if avg_elbo <= self.best_loss:
            self.best_loss = avg_elbo
            logging.info('New best validation loss! (%.3f)' % self.best_loss)
            best_file = os.path.join(self.log_dir, 'model_best_loss.ckpt')
            self.saver_best_loss.save(self.sess, best_file, global_step=global_step)

        if avg_ged <= self.best_ged:
            self.best_ged = avg_ged
            logging.info('New best GED score! (%.3f)' % self.best_ged)
            best_file = os.path.join(self.log_dir, 'model_best_ged.ckpt')
            self.saver_best_ged.save(self.sess, best_file, global_step=global_step)

        if avg_ncc >= self.best_ncc:
            self.best_ncc = avg_ncc
            logging.info('New best NCC score! (%.3f)' % self.best_ncc)
            best_file = os.path.join(self.log_dir, 'model_best_ncc.ckpt')
            self.saver_best_ncc.save(self.sess, best_file, global_step=global_step)

        # Create Validation Summary feed dict
        z_prior_list = self.generate_prior_samples(x_in=val_x)
        val_summary_feed_dict = {i: d for i, d in zip(self.z_list_gen, z_prior_list)} # this is for prior samples
        val_summary_feed_dict[self.x_for_gen] = val_x

        # Fill placeholders for all losses
        for loss_key, loss_val in zip(self.loss_dict.keys(), val_losses_out):
            # The detour over loss_dict.keys() is necessary because val_losses_out is sorted in the same
            # way as loss_dict. Same for the training below.
            loss_pl = self.validation_loss_pl_dict[loss_key]
            val_summary_feed_dict[loss_pl] = loss_val

        # Fill placeholders for validation Dice
        val_summary_feed_dict[self.val_tot_dice_score] = avg_dice
        val_summary_feed_dict[self.val_mean_dice_score] = np.mean(per_structure_dice)
        val_summary_feed_dict[self.val_elbo] = avg_elbo
        val_summary_feed_dict[self.val_ged] = avg_ged
        val_summary_feed_dict[self.val_ncc] = np.squeeze(avg_ncc)

        for ii in range(self.exp_config.nlabels):
            val_summary_feed_dict[self.val_lbl_dice_scores[ii]] = per_structure_dice[ii]

        val_summary_feed_dict[self.x_inp] = val_x
        val_summary_feed_dict[self.s_inp] = val_s
        val_summary_feed_dict[self.training_pl] = False

        val_summary_msg = self.sess.run(self.val_summary, feed_dict=val_summary_feed_dict)
        self.summary_writer.add_summary(val_summary_msg, global_step)

        # Create train Summary feed dict
        train_summary_feed_dict = {}
        for loss_key, loss_val in zip(self.loss_dict.keys(), train_losses_out):
            loss_pl = self.train_loss_pl_dict[loss_key]
            train_summary_feed_dict[loss_pl] = loss_val
        train_summary_feed_dict[self.training_pl] = False

        train_summary_msg = self.sess.run(self.train_summary,
                                          feed_dict=train_summary_feed_dict
                                          )
        self.summary_writer.add_summary(train_summary_msg, global_step)


    def _make_tensorboard_summaries(self):

        def create_im_summary(img, name, rescale_mode, batch_size=self.exp_config.batch_size):

            if tfutils.tfndims(img) == 3:
                img_disp = tf.expand_dims(img, axis=-1)
            elif tfutils.tfndims(img) == 4:
                img_disp = img
            else:
                raise ValueError("Unexpected tensor ndim: %d" % tfutils.tfndims(img))

            nlabels = self.exp_config.nlabels if rescale_mode == 'labelmap' else None
            return tf.summary.image(name, tfutils.put_kernels_on_grid(img_disp, batch_size=batch_size, rescale_mode=rescale_mode, nlabels=nlabels))


        tf.summary.scalar('batch_total_loss', self.loss_tot)
        tf.summary.scalar('learning_rate', self.lr_pl)

        for ii, (mu, sigma) in enumerate(zip(self.mu_list, self.sigma_list)):
            tf.summary.scalar('average_mu_lvl%d' % ii, tf.reduce_mean(mu))
            tf.summary.scalar('average_sigma_lvl%d' % ii, tf.reduce_mean(sigma))
            tf.summary.scalar('average_prior_mu_lvl%d' % ii, tf.reduce_mean(self.prior_mu_list[ii]))
            tf.summary.scalar('average_prior_sigma_lvl%d' % ii, tf.reduce_mean(self.prior_sigma_list[ii]))

        if self.exp_config.do_image_summaries:

            create_im_summary(self.x_inp, 'train_x_inp', rescale_mode='standardize')
            create_im_summary(self.s_inp, 'train_s_inp', rescale_mode='labelmap')
            create_im_summary(tf.argmax(self.s_out, axis=-1), 'train_s_out', rescale_mode='labelmap')

            for ii in range(self.exp_config.latent_levels):
                create_im_summary(tf.argmax(self.s_out_list[ii], axis=-1), 'train_s_out_list_%d' % ii, rescale_mode='labelmap')
                create_im_summary(tf.argmax(self.s_accum[ii], axis=-1), 'train_s_accum_list_%d' % ii, rescale_mode='labelmap')


        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()

        # Validation summaries
        self.validation_loss_pl_dict= {}
        val_summary_list = []
        for loss_name in self.loss_dict.keys():
            self.validation_loss_pl_dict[loss_name] = tf.placeholder(tf.float32, shape=[], name='val_%s' % loss_name)
            val_summary_list.append(tf.summary.scalar('val_batch_%s' % loss_name, self.validation_loss_pl_dict[loss_name]))

        self.val_summary = tf.summary.merge(val_summary_list)

        if self.exp_config.do_image_summaries:

            # Validation reconstructions

            val_img_sum = []

            val_img_sum.append(create_im_summary(self.x_inp, 'val_x_inp', rescale_mode='standardize'))
            val_img_sum.append(create_im_summary(self.s_inp, 'val_s_inp', rescale_mode='labelmap'))
            val_img_sum.append(create_im_summary(tf.argmax(self.s_out, axis=-1), 'val_s_out', rescale_mode='labelmap'))

            for ii in range(self.exp_config.latent_levels):

                val_img_sum.append(create_im_summary(tf.argmax(self.s_out_list[ii], axis=-1), 'val_s_out_list_%d' % ii, rescale_mode='labelmap'))
                val_img_sum.append(create_im_summary(tf.argmax(self.s_accum[ii], axis=-1), 'val_s_accum_list_%d' % ii, rescale_mode='labelmap'))

            self.x_for_gen = tf.placeholder(tf.float32, shape=self.x_inp.shape)
            self.z_list_gen = []
            for z in self.z_list:
                self.z_list_gen.append(tf.placeholder(tf.float32, shape=z.shape))

            s_from_prior = tf.argmax(self.s_out_eval, axis=-1)

            val_img_sum.append(create_im_summary(s_from_prior, 'generated_seg', rescale_mode='labelmap'))
            val_img_sum.append(create_im_summary(self.x_for_gen, 'generated_x_in', rescale_mode='standardize'))

            self.val_summary = tf.summary.merge([self.val_summary, val_img_sum])

        # Val Dice summaries

        self.val_tot_dice_score = tf.placeholder(tf.float32, shape=[], name='val_dice_total_score')
        val_tot_dice_summary = tf.summary.scalar('validation_dice_tot_score', self.val_tot_dice_score)

        self.val_mean_dice_score = tf.placeholder(tf.float32, shape=[], name='val_dice_mean_score')
        val_mean_dice_summary = tf.summary.scalar('validation_dice_mean_score', self.val_mean_dice_score)

        self.val_elbo = tf.placeholder(tf.float32, shape=[], name='val_elbo')
        val_elbo_summary = tf.summary.scalar('validation_neg_elbo', self.val_elbo)

        self.val_ged = tf.placeholder(tf.float32, shape=[], name='val_ged')
        val_ged_summary = tf.summary.scalar('validation_GED', self.val_ged)

        self.val_ncc = tf.placeholder(tf.float32, shape=[], name='val_ncc')
        val_ncc_summary = tf.summary.scalar('validation_NCC', self.val_ncc)


        self.val_lbl_dice_scores = []
        val_lbl_dice_summaries = []
        for ii in range(self.exp_config.nlabels):
            curr_pl = tf.placeholder(tf.float32, shape=[], name='validation_dice_lbl_%d' % ii)
            self.val_lbl_dice_scores.append(curr_pl)
            val_lbl_dice_summaries.append(tf.summary.scalar('validation_dice_lbl_%d' % ii, curr_pl))

        self.val_summary = tf.summary.merge([self.val_summary,
                                             val_tot_dice_summary,
                                             val_mean_dice_summary,
                                             val_lbl_dice_summaries,
                                             val_elbo_summary,
                                             val_ged_summary,
                                             val_ncc_summary])

        # Train summaries
        self.train_loss_pl_dict= {}
        train_summary_list = []
        for loss_name in self.loss_dict.keys():
            self.train_loss_pl_dict[loss_name] = tf.placeholder(tf.float32, shape=[], name='val_%s' % loss_name)
            train_summary_list.append(tf.summary.scalar('train_batch_%s' % loss_name, self.train_loss_pl_dict[loss_name]))

        self.train_summary = tf.summary.merge(train_summary_list)


    def _setup_log_dir_and_continue_mode(self):

        # Default values
        self.log_dir = os.path.join(sys_config.log_root, self.exp_config.log_dir_name, self.exp_config.experiment_name)
        self.init_checkpoint_path = None
        self.continue_run = False
        self.init_step = 0

        # If a checkpoint file already exists enable continue mode
        if tf.gfile.Exists(self.log_dir):
            init_checkpoint_path = tfutils.get_latest_model_checkpoint_path(self.log_dir, 'model.ckpt')
            if init_checkpoint_path is not False:

                self.init_checkpoint_path = init_checkpoint_path
                self.continue_run = True
                self.init_step = int(self.init_checkpoint_path.split('/')[-1].split('-')[-1])
                self.log_dir += '_cont'

                logging.info('--------------------------- Continuing previous run --------------------------------')
                logging.info('Checkpoint path: %s' % self.init_checkpoint_path)
                logging.info('Latest step was: %d' % self.init_step)
                logging.info('------------------------------------------------------------------------------------')

        tf.gfile.MakeDirs(self.log_dir)
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # Copy experiment config file to log_dir for future reference
        # shutil.copy(self.exp_config.__file__, self.log_dir)
        # logging.info('!!!! Copied exp_config file to experiment folder !!!!')