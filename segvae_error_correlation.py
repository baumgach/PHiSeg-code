
import glob
import logging
import os
from importlib.machinery import SourceFileLoader

import numpy as np

import config.system as sys_config
import utils
from data.data_switch import data_switch
from phiseg.phiseg_model import segvae
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main(model_path, exp_config):

    # Make and restore vagan model
    segvae_model = segvae(exp_config=exp_config)
    segvae_model.load_weights(model_path, type='latest')

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    ncc_list = []

    for ii, batch in enumerate(data.test.iterate_batches(1)):

        if ii % 10 == 0:
            print('Progress: %d' % ii)

        x_b, s_b = batch

        s_m, s_v, s_e = segvae_model.predict_mean_variance_and_error_maps(s_b, x_b, num_samples=100)

        ncc_list.append(utils.ncc(s_v, s_e))

    ncc_arr = np.asarray(ncc_list)

    ncc_mean = np.mean(ncc_arr, axis=0)
    ncc_std = np.std(ncc_arr, axis=0)

    print('NCC mean: %.4f', ncc_mean)
    print('NCC std: %.4f', ncc_std)


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

    main(model_path, exp_config=exp_config)
