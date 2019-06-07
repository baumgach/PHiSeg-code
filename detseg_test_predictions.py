# Get classification metrics for a trained classifier model
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from medpy.metric import dc, assd, hd

import config.system as sys_config
from detseg.model_segmenter import segmenter as segmenter
import utils

if not sys_config.running_on_gpu_host:
    import matplotlib.pyplot as plt

import logging
from data.data_switch import data_switch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main(model_path, exp_config, do_plots=False):

    # Get Data
    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    # Make and restore vagan model
    segmenter_model = segmenter(exp_config=exp_config, data=data, fixed_batch_size=1)  # CRF model requires fixed batch size
    segmenter_model.load_weights(model_path, type='best_dice')

    # Run predictions in an endless loop
    dice_list = []
    assd_list = []
    hd_list = []

    logging.info('WARNING: Adding motion corruption!')

    for ii, batch in enumerate(data.test.iterate_batches(1)):

        if ii % 100 == 0:
            logging.info("Progress: %d" % ii)

        x, y = batch

        # Adding motion corrpution
        # x = utils.add_motion_artefacts(np.squeeze(x), 15)
        # x = x.reshape([1] + list(exp_config.image_size) + [1])
        # Add box corruption
        # x[:, 192 // 2 - 20:192 // 2 + 20, 192 // 2 - 5:192 // 2 + 5, :] = 0


        y_ = segmenter_model.predict(x)[0]

        per_lbl_dice = []
        per_lbl_assd = []
        per_lbl_hd = []
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
                # per_lbl_assd.append(0)
                # per_lbl_hd.append(0)
            elif np.sum(binary_pred) > 0 and np.sum(binary_gt) == 0 or np.sum(binary_pred) == 0 and np.sum(binary_gt) > 0:
                logging.warning('Structure missing in either GT (x)or prediction. ASSD and HD will not be accurate.')
                per_lbl_dice.append(0)
                # per_lbl_assd.append(1)
                # per_lbl_hd.append(1)
            else:
                per_lbl_dice.append(dc(binary_pred, binary_gt))
                # per_lbl_assd.append(assd(binary_pred, binary_gt))
                # per_lbl_hd.append(hd(binary_pred, binary_gt))

        dice_list.append(per_lbl_dice)
        assd_list.append(per_lbl_assd)
        hd_list.append(per_lbl_hd)

        per_pixel_preds.append(y_.flatten())
        per_pixel_gts.append(y.flatten())

    dice_arr = np.asarray(dice_list)
    assd_arr = np.asarray(assd_list)
    hd_arr = np.asarray(hd_list)

    mean_per_lbl_dice = dice_arr.mean(axis=0)
    mean_per_lbl_assd = assd_arr.mean(axis=0)
    mean_per_lbl_hd = hd_arr.mean(axis=0)

    logging.info('Dice')
    logging.info(mean_per_lbl_dice)
    logging.info(np.mean(mean_per_lbl_dice))
    logging.info('foreground mean: %f' % (np.mean(mean_per_lbl_dice[1:])))
    np.savez(os.path.join(model_path, 'dice.npz'), dice_arr)
    # logging.info('ASSD')
    # logging.info(structures_dict)
    # logging.info(mean_per_lbl_assd)
    # logging.info(np.mean(mean_per_lbl_assd))
    # logging.info('HD')
    # logging.info(structures_dict)
    # logging.info(mean_per_lbl_hd)
    # logging.info(np.mean(mean_per_lbl_hd))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a network on the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config, do_plots=False)

