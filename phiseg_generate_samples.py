
import glob
import logging
import os
from importlib.machinery import SourceFileLoader
import cv2
import argparse

import numpy as np

import config.system as sys_config
import utils
from data.data_switch import data_switch
from phiseg.phiseg_model import segvae

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

model_selection = 'best_ged'

def preproc_image(x, nlabels=None):

    x_b = np.squeeze(x)

    ims = x_b.shape[:2]

    if nlabels:
        x_b = np.uint8((x_b / (nlabels)) * 255)  # not nlabels - 1 because I prefer gray over white
    else:
        x_b = utils.convert_to_uint8(x_b)

    # x_b = cv2.cvtColor(np.squeeze(x_b), cv2.COLOR_GRAY2BGR)
    # x_b = utils.histogram_equalization(x_b)
    x_b = utils.resize_image(x_b, (2 * ims[0], 2 * ims[1]), interp=cv2.INTER_NEAREST)

    # ims_n = x_b.shape[:2]
    # x_b = x_b[ims_n[0]//4:3*ims_n[0]//4, ims_n[1]//4: 3*ims_n[1]//4,...]
    return x_b


def generate_error_maps(sample_arr, gt_arr):

    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):


        log_samples = np.log(m_samp + eps)
        return -1.0*np.sum(m_gt*log_samples, axis=-1)

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy_avg = np.mean(np.mean(E_sy_arr, axis=1), axis=0)

    E_yy_arr = np.zeros((M,M, sX, sY))
    for j in range(M):
        for i in range(M):
            E_yy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_yy_avg = np.mean(np.mean(E_yy_arr, axis=1), axis=0)

    return E_ss, E_sy_avg, E_yy_avg



def main(model_path, exp_config):

    # Make and restore vagan model
    segvae_model = segvae(exp_config=exp_config)
    segvae_model.load_weights(model_path, type=model_selection)

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    N = data.test.images.shape[0]

    n_images = 16
    n_samples = 16

    # indices = np.arange(N)
    # sample_inds = np.random.choice(indices, n_images)
    sample_inds = [165, 280, 213]  # <-- prostate
    # sample_inds = [1551] #[907, 1296, 1551]  # <-- LIDC

    for ii in sample_inds:

        print('------- Processing image %d -------' % ii)

        outfolder = os.path.join(model_path, 'samples_%s' % model_selection, str(ii))
        utils.makefolder(outfolder)

        x_b = data.test.images[ii, ...].reshape([1] + list(exp_config.image_size))
        s_b = data.test.labels[ii, ...]

        if np.sum(s_b) < 10:
            print('WARNING: skipping cases with no structures')
            continue

        s_b_r = utils.convert_batch_to_onehot(s_b.transpose((2, 0, 1)), exp_config.nlabels)

        print('Plotting input image')
        plt.figure()
        x_b_d = preproc_image(x_b)
        plt.imshow(x_b_d, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(outfolder, 'input_img_%d.png' % ii),bbox_inches='tight')

        print('Generating 100 samples')
        s_p_list = []
        for kk in range(100):
            s_p_list.append(segvae_model.predict_segmentation_sample(x_b, return_softmax=True))
        s_p_arr = np.squeeze(np.asarray(s_p_list))


        print('Plotting %d of those samples' % n_samples)
        for jj in range(n_samples):

            s_p_sm = s_p_arr[jj,...]
            s_p_am = np.argmax(s_p_sm, axis=-1)

            plt.figure()
            s_p_d = preproc_image(s_p_am, nlabels=exp_config.nlabels)
            plt.imshow(s_p_d, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(outfolder, 'sample_img_%d_samp_%d.png' % (ii,jj)),bbox_inches='tight')

        print('Plotting ground-truths masks')
        for jj in range(s_b_r.shape[0]):

            s_b_sm = s_b_r[jj,...]
            s_b_am = np.argmax(s_b_sm, axis=-1)

            plt.figure()
            s_p_d = preproc_image(s_b_am, nlabels=exp_config.nlabels)
            plt.imshow(s_p_d, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(outfolder, 'gt_img_%d_samp_%d.png' % (ii,jj)),bbox_inches='tight')

        print('Generating error masks')
        E_ss, E_sy_avg, E_yy_avg = generate_error_maps(s_p_arr, s_b_r)

        print('Plotting them')
        plt.figure()
        plt.imshow(preproc_image(E_ss))
        plt.axis('off')
        plt.savefig(os.path.join(outfolder, 'E_ss_%d.png' % ii), bbox_inches='tight')

        print('Plotting them')
        plt.figure()
        plt.imshow(preproc_image(np.log(E_ss)))
        plt.axis('off')
        plt.savefig(os.path.join(outfolder, 'log_E_ss_%d.png' % ii), bbox_inches='tight')


        plt.figure()
        plt.imshow(preproc_image(E_sy_avg))
        plt.axis('off')
        plt.savefig(os.path.join(outfolder, 'E_sy_avg_%d_.png' % ii), bbox_inches='tight')

        plt.figure()
        plt.imshow(preproc_image(E_yy_avg))
        plt.axis('off')
        plt.savefig(os.path.join(outfolder, 'E_yy_avg_%d_.png' % ii), bbox_inches='tight')

        plt.close('all')

    # plt.show()

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
