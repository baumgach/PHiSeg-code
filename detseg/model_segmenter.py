# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from skimage import transform  # used for CAM saliency

from tfwrapper import losses
from tfwrapper import utils as tf_utils
import config.system as sys_config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()

class segmenter:

    def __init__(self, exp_config, data, fixed_batch_size=None):

        self.exp_config = exp_config
        self.data = data

        self.nlabels = exp_config.nlabels

        self.image_tensor_shape = [fixed_batch_size] + list(exp_config.image_size) + [1]
        self.labels_tensor_shape = [fixed_batch_size] + list(exp_config.image_size)

        self.x_pl = tf.placeholder(tf.float32, shape=self.image_tensor_shape, name='images')
        self.y_pl = tf.placeholder(tf.uint8, shape=self.labels_tensor_shape, name='labels')

        self.lr_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.training_pl = tf.placeholder(tf.bool, shape=[], name='training_time')

        self.l_pl_ = exp_config.network(self.x_pl,
                                        nlabels=self.nlabels,
                                        training=self.training_pl,
                                        n0=self.exp_config.n0,
                                        norm=exp_config.normalisation,
                                        resolution_levels=exp_config.resolution_levels)

        self.y_pl_ = tf.nn.softmax(self.l_pl_)
        self.p_pl_ = tf.cast(tf.argmax(self.y_pl_, axis=-1), tf.int32)

        # Add to the Graph the Ops for loss calculation.

        self.task_loss = self.loss()
        self.weights_norm = self.weight_norm()

        self.total_loss = self.task_loss + self.exp_config.weight_decay * self.weights_norm

        self.global_step = tf.train.get_or_create_global_step()  # Used in batch renormalisation

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = self._get_optimizer()
            self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

        self.global_step = tf.train.get_or_create_global_step()
        self.increase_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))

        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        self.saver_best_xent = tf.train.Saver(max_to_keep=2)
        self.saver_best_dice = tf.train.Saver(max_to_keep=2)

        # Settings to optimize GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        # For evaluation
        self.eval_dice_per_structure = losses.get_dice(self.l_pl_, tf.one_hot(self.y_pl, depth=self.nlabels), sum_over_labels=False)
        self.eval_dice_per_image = losses.get_dice(self.l_pl_, tf.one_hot(self.y_pl, depth=self.nlabels), sum_over_labels=True)

        # Create a session for running Ops on the Graph.
        self.sess = tf.Session(config=config)

    def loss(self):

        y_for_loss = tf.one_hot(self.y_pl, depth=self.nlabels)

        if self.exp_config.loss_type == 'crossentropy':
            task_loss = losses.cross_entropy_loss(logits=self.l_pl_, labels=y_for_loss)
        elif self.exp_config.loss_type == 'dice_micro':
            task_loss = losses.dice_loss(logits=self.l_pl_, labels=y_for_loss, mode='micro')
        elif self.exp_config.loss_type == 'dice_macro':
            task_loss = losses.dice_loss(logits=self.l_pl_, labels=y_for_loss, mode='macro')
        elif self.exp_config.loss_type == 'dice_macro_robust':
            task_loss = losses.dice_loss(logits=self.l_pl_, labels=y_for_loss, mode='macro_robust')
        else:
            raise ValueError("Unknown loss_type in exp_config: '%s'" % self.exp_config.loss_type)

        return task_loss

    def weight_norm(self):

        weights_norm = tf.reduce_sum(
            input_tensor= tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )

        return weights_norm


    def train(self):

        # Sort out proper logging
        self._setup_log_dir_and_continue_mode()

        # Create tensorboard summaries
        self._make_tensorboard_summaries()

        # self.curr_lr = self.exp_config.learning_rate
        # schedule_lr = True if self.exp_config.divide_lr_frequency is not None else False

        logging.info('===== RUNNING EXPERIMENT ========')
        logging.info(self.exp_config.experiment_name)
        logging.info('=================================')

        # initialise all weights etc..
        self.sess.run(tf.global_variables_initializer())

        # Restore session if there is one
        if self.continue_run:
            self.saver.restore(self.sess, self.init_checkpoint_path)

        logging.info('Starting training:')

        best_val = np.inf
        best_dice_score = 0

        for step in range(self.init_step, self.exp_config.max_iterations):

            # Get current learning rate from lr_dict
            lr_key, _ = utils.find_floor_in_list(self.exp_config.lr_schedule_dict.keys(), step)
            lr = self.exp_config.lr_schedule_dict[lr_key]

            x_b, y_b = self.data.train.next_batch(self.exp_config.batch_size)

            feed_dict = {self.x_pl: x_b,  # dummy variables will be replaced in optimizer
                         self.y_pl: y_b,
                         self.training_pl: True,
                         self.lr_pl: lr}

            start_time = time.time()

            loss_value, _ = self.sess.run([self.total_loss, self.train_op], feed_dict=feed_dict)

            elapsed_time = time.time() - start_time


            ###  Tensorboard updates, Model Saving, and Validation

            # Update tensorboard
            if step % 5 == 0:

                logging.info('Step %d: loss = %.2f (One update step took %.3f sec)' % (step, loss_value, elapsed_time))

                summary_str = self.sess.run(self.summary, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()

            # Do training evaluation
            if (step + 1) % self.exp_config.train_eval_frequency == 0:

                # Evaluate against the training set
                logging.info('Training Data Eval:')
                self._do_validation(self.data.train,
                                    self.train_summary,
                                    self.train_error,
                                    self.train_tot_dice_score,
                                    self.train_mean_dice_score,
                                    self.train_lbl_dice_scores)

            # Do validation set evaluation
            if (step + 1) % self.exp_config.val_eval_frequency == 0:

                checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_file, global_step=step)

                # Evaluate against the validation set.
                logging.info('Validation Data Eval:')

                val_loss, val_dice = self._do_validation(self.data.validation,
                                                         self.val_summary,
                                                         self.val_error,
                                                         self.val_tot_dice_score,
                                                         self.val_mean_dice_score,
                                                         self.val_lbl_dice_scores)

                if val_dice >= best_dice_score:
                    best_dice_score = val_dice
                    best_file = os.path.join(self.log_dir, 'model_best_dice.ckpt')
                    self.saver_best_dice.save(self.sess, best_file, global_step=step)
                    logging.info( 'Found new best Dice score on validation set! - %f -  Saving model_best_dice.ckpt' % val_dice)

                if val_loss < best_val:
                    best_val = val_loss
                    best_file = os.path.join(self.log_dir, 'model_best_xent.ckpt')
                    self.saver_best_xent.save(self.sess, best_file, global_step=step)
                    logging.info('Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)

            self.sess.run(self.increase_global_step)


    def load_weights(self, log_dir=None, type='latest', **kwargs):

        if not log_dir:
            log_dir = self.log_dir

        if type=='latest':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
        elif type=='best_dice':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model_best_dice.ckpt')
        elif type=='best_xent':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model_best_xent.ckpt')
        elif type=='iter':
            assert 'iteration' in kwargs, "argument 'iteration' must be provided for type='iter'"
            iteration = kwargs['iteration']
            init_checkpoint_path = os.path.join(log_dir, 'model.ckpt-%d' % iteration)
        else:
            raise ValueError('Argument type=%s is unknown. type can be latest/best_wasserstein/iter.' % type)

        self.saver.restore(self.sess, init_checkpoint_path)


    def predict(self, images):

        prediction, softmax = self.sess.run([self.p_pl_, self.y_pl_],
                                            feed_dict={self.x_pl: images, self.training_pl: False})

        return prediction, softmax

    ### HELPER FUNCTIONS ###################################################################################

    def _make_tensorboard_summaries(self):

        ### Batch-wise summaries

        tf.summary.scalar('learning_rate', self.lr_pl)

        tf.summary.scalar('task_loss', self.task_loss)
        tf.summary.scalar('weights_norm', self.weights_norm)
        tf.summary.scalar('total_loss', self.total_loss)

        def _image_summaries(prefix, x, y, y_gt):

            if len(self.image_tensor_shape) == 5:
                data_dimension = 3
            elif len(self.image_tensor_shape) == 4:
                data_dimension = 2
            else:
                raise ValueError('Invalid image dimensions')

            if data_dimension == 3:
                y_disp = tf.expand_dims(y[:, :, :, self.exp_config.tensorboard_slice], axis=-1)
                y_gt_disp = tf.expand_dims(y_gt[:, :, :, self.exp_config.tensorboard_slice], axis=-1)
                x_disp = x[:, :, :, self.exp_config.tensorboard_slice, :]
            else:
                y_disp = tf.expand_dims(y, axis=-1)
                y_gt_disp = tf.expand_dims(y_gt, axis=-1)
                x_disp = x

            sum_y = tf.summary.image('%s_mask_predicted' % prefix, tf_utils.put_kernels_on_grid(
                y_disp,
                batch_size=self.exp_config.batch_size,
                mode='mask',
                nlabels=self.exp_config.nlabels))
            sum_y_gt = tf.summary.image('%s_mask_groundtruth' % prefix, tf_utils.put_kernels_on_grid(
                y_gt_disp,
                batch_size=self.exp_config.batch_size,
                mode='mask',
                nlabels=self.exp_config.nlabels))
            sum_x = tf.summary.image('%s_input_image' % prefix, tf_utils.put_kernels_on_grid(
                x_disp,
                min_int=None,
                max_int=None,
                batch_size=self.exp_config.batch_size,
                mode='image'))

            return tf.summary.merge([sum_y, sum_y_gt, sum_x])

        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()

        ### Validation summaries

        self.val_error = tf.placeholder(tf.float32, shape=[], name='val_task_loss')
        val_error_summary = tf.summary.scalar('validation_task_loss', self.val_error)

        # Note: Total dice is the Dice over all pixels of an image
        #       Mean Dice is the mean of the per label dices, which is not affected by class imbalance

        self.val_tot_dice_score = tf.placeholder(tf.float32, shape=[], name='val_dice_total_score')
        val_tot_dice_summary = tf.summary.scalar('validation_dice_tot_score', self.val_tot_dice_score)

        self.val_mean_dice_score = tf.placeholder(tf.float32, shape=[], name='val_dice_mean_score')
        val_mean_dice_summary = tf.summary.scalar('validation_dice_mean_score', self.val_mean_dice_score)

        self.val_lbl_dice_scores = []
        val_lbl_dice_summaries = []
        for ii in range(self.nlabels):
            curr_pl = tf.placeholder(tf.float32, shape=[], name='val_dice_lbl_%d' % ii)
            self.val_lbl_dice_scores.append(curr_pl)
            val_lbl_dice_summaries.append(tf.summary.scalar('validation_dice_lbl_%d' % ii, curr_pl))

        val_image_summary = _image_summaries('validation', self.x_pl, self.p_pl_, self.y_pl)

        self.val_summary = tf.summary.merge([val_error_summary,
                                             val_tot_dice_summary,
                                             val_mean_dice_summary,
                                             val_image_summary] + val_lbl_dice_summaries)

        ### Train summaries

        self.train_error = tf.placeholder(tf.float32, shape=[], name='train_task_loss')
        train_error_summary = tf.summary.scalar('training_task_loss', self.train_error)

        self.train_tot_dice_score = tf.placeholder(tf.float32, shape=[], name='train_dice_tot_score')
        train_tot_dice_summary = tf.summary.scalar('train_dice_tot_score', self.train_tot_dice_score)

        self.train_mean_dice_score = tf.placeholder(tf.float32, shape=[], name='train_dice_mean_score')
        train_mean_dice_summary = tf.summary.scalar('train_dice_mean_score', self.train_mean_dice_score)

        self.train_lbl_dice_scores = []
        train_lbl_dice_summaries = []
        for ii in range(self.nlabels):
            curr_pl = tf.placeholder(tf.float32, shape=[], name='train_dice_lbl_%d' % ii)
            self.train_lbl_dice_scores.append(curr_pl)
            train_lbl_dice_summaries.append(tf.summary.scalar('train_dice_lbl_%d' % ii, curr_pl))

        train_image_summary = _image_summaries('train', self.x_pl, self.p_pl_, self.y_pl)

        self.train_summary = tf.summary.merge([train_error_summary,
                                               train_tot_dice_summary,
                                               train_mean_dice_summary,
                                               train_image_summary] + train_lbl_dice_summaries)


    def _get_optimizer(self):

        if self.exp_config.optimizer_handle == tf.train.AdamOptimizer:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl,
                                                    beta1=self.exp_config.beta1,
                                                    beta2=self.exp_config.beta2)
        if self.exp_config.momentum is not None:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl,
                                                    momentum=self.exp_config.momentum)
        else:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl)


    def _setup_log_dir_and_continue_mode(self):

        # Default values
        self.log_dir = os.path.join(sys_config.log_root, self.exp_config.log_dir_name, self.exp_config.experiment_name)
        self.init_checkpoint_path = None
        self.continue_run = False
        self.init_step = 0

        # If a checkpoint file already exists enable continue mode
        if tf.gfile.Exists(self.log_dir):
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(self.log_dir, 'model.ckpt')
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
        shutil.copy(self.exp_config.__file__, self.log_dir)


    def _eval_predict(self, images, labels):

        prediction, loss, dice_per_img_and_lbl, dice_per_img = self.sess.run([self.p_pl_,
                                                                              self.total_loss,
                                                                              self.eval_dice_per_structure,
                                                                              self.eval_dice_per_image],
                                                                             feed_dict={self.x_pl: images,
                                                                             self.y_pl: labels,
                                                                             self.training_pl: False})

        # We are getting per_image and per_structure dice separately because the average of per_structure
        # will be wrong for labels that do not appear in an image.

        return prediction, loss, dice_per_img_and_lbl, dice_per_img


    def _do_validation(self, data_handle, summary, tot_loss_pl, tot_dice_pl, mean_dice_pl, lbl_dice_pl_list):

        diag_loss_ii = 0
        num_batches = 0
        all_dice_per_img_and_lbl = []
        all_dice_per_img = []


        for batch in data_handle.iterate_batches(self.exp_config.batch_size):

            x, y = batch

            # Skip incomplete batches
            if y.shape[0] < self.exp_config.batch_size:
                continue

            c_d_preds, c_d_loss, dice_per_img_and_lbl, dice_per_img = self._eval_predict(x, y)

            num_batches += 1
            diag_loss_ii += c_d_loss

            all_dice_per_img_and_lbl.append(dice_per_img_and_lbl)
            all_dice_per_img.append(dice_per_img)

        avg_loss = (diag_loss_ii / num_batches)

        dice_per_lbl_array = np.asarray(all_dice_per_img_and_lbl).reshape((-1, self.nlabels))

        per_structure_dice = np.mean(dice_per_lbl_array, axis=0)

        dice_array = np.asarray(all_dice_per_img).flatten()

        avg_dice = np.mean(dice_array)

        ### Update Tensorboard
        x, y = data_handle.next_batch(self.exp_config.batch_size)
        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_pl] = x
        feed_dict[self.y_pl] = y
        feed_dict[tot_loss_pl] = avg_loss
        feed_dict[tot_dice_pl] = avg_dice
        feed_dict[mean_dice_pl] = np.mean(per_structure_dice)
        for ii in range(self.nlabels):
            feed_dict[lbl_dice_pl_list[ii]] = per_structure_dice[ii]

        summary_msg = self.sess.run(summary, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_msg, global_step=self.sess.run(tf.train.get_global_step()))

        ### Output logs
        # Note: Total dice is the Dice over all pixels of an image
        #       Mean Dice is the mean of the per label dices, which is not affected by class imbalance
        logging.info('  Average loss: %0.04f' % avg_loss)
        logging.info('  Total Dice: %0.04f' % avg_dice)
        logging.info('  Mean Dice: %0.04f' % np.mean(per_structure_dice))
        for ii in range(self.nlabels):
            logging.info('  Dice lbl %d: %0.04f' % (ii, per_structure_dice[ii]))
        logging.info('---')

        return avg_loss, np.mean(per_structure_dice)



