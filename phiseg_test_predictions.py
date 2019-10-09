# Get classification metrics for a trained classifier model
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

from phiseg.model_zoo import likelihoods
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from medpy.metric import dc, assd, hd

import config.system as sys_config
from phiseg.phiseg_model import phiseg
import utils

if not sys_config.running_on_gpu_host:
    import matplotlib.pyplot as plt

import logging
from data.data_switch import data_switch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}

model_selection = 'best_dice'

def main(model_path, exp_config, do_plots=False):

    # Get Data
    segvae_model = phiseg(exp_config=exp_config)
    segvae_model.load_weights(model_path, type=model_selection)

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    # Run predictions in an endless loop
    dice_list = []

    num_samples = 1 if exp_config.likelihood is likelihoods.det_unet2D else 100

    for ii, batch in enumerate(data.test.iterate_batches(1)):

        if ii % 10 == 0:
            logging.info("Progress: %d" % ii)

        # print(ii)

        x, y = batch

        # Adding motion corrpution
        # x = utils.add_motion_artefacts(np.squeeze(x), 15)
        # x = x.reshape([1] + list(exp_config.image_size))

        # Add box corruption
        # x[:, 192 // 2 - 20:192 // 2 + 20, 192 // 2 - 5:192 // 2 + 5, :] = 0

        y_ = np.squeeze(segvae_model.predict(x, num_samples=num_samples))

        per_lbl_dice = []
        per_pixel_preds = []
        per_pixel_gts = []

        if do_plots and not sys_config.running_on_gpu_host:
            fig = plt.figure()
            fig.add_subplot(131)
            plt.imshow(np.squeeze(x), cmap='gray')
            fig.add_subplot(132)
            plt.imshow(np.squeeze(y_))
            fig.add_subplot(133)
            plt.imshow(np.squeeze(y))
            plt.show()

        for lbl in range(exp_config.nlabels):

            binary_pred = (y_ == lbl) * 1
            binary_gt = (y == lbl) * 1

            if np.sum(binary_gt) == 0 and np.sum(binary_pred) == 0:
                per_lbl_dice.append(1)
            elif np.sum(binary_pred) > 0 and np.sum(binary_gt) == 0 or np.sum(binary_pred) == 0 and np.sum(binary_gt) > 0:
                logging.warning('Structure missing in either GT (x)or prediction. ASSD and HD will not be accurate.')
                per_lbl_dice.append(0)
            else:
                per_lbl_dice.append(dc(binary_pred, binary_gt))

        dice_list.append(per_lbl_dice)

        per_pixel_preds.append(y_.flatten())
        per_pixel_gts.append(y.flatten())

    dice_arr = np.asarray(dice_list)

    mean_per_lbl_dice = dice_arr.mean(axis=0)

    logging.info('Dice')
    logging.info(mean_per_lbl_dice)
    logging.info(np.mean(mean_per_lbl_dice))
    logging.info('foreground mean: %f' % (np.mean(mean_per_lbl_dice[1:])))

    np.savez(os.path.join(model_path, 'dice_%s.npz' % model_selection), dice_arr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a network on the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = args.EXP_PATH
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config, do_plots=False)

