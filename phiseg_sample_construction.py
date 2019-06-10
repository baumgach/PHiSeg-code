
import glob
import logging
import os
from importlib.machinery import SourceFileLoader
import cv2

import numpy as np

import config.system as sys_config
import utils
from data.data_switch import data_switch
from phiseg.phiseg_model import segvae

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import matplotlib.pyplot as plt

import itertools
def findsubsets(S,m):
    return list(itertools.combinations(S, m))

def main(model_path, exp_config):

    # Make and restore vagan model
    segvae_model = segvae(exp_config=exp_config)
    segvae_model.load_weights(model_path, type='best_dice')

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    outfolder = '/home/baumgach/Reports/ETH/MICCAI2019_segvae/raw_figures'

    ims = exp_config.image_size

    # x_b, s_b = data.test.next_batch(1)

    # heart 100
    # prostate 165
    index = 165 # 100 is a normal image, 15 is a very good slice
    x_b = data.test.images[index, ...].reshape([1] + list(exp_config.image_size))
    if exp_config.data_identifier == 'lidc':
        s_b = data.test.labels[index, ...]
        if np.sum(s_b[..., 0]) > 0:
            s_b = s_b[..., 0]
        elif np.sum(s_b[..., 1]) > 0:
            s_b = s_b[..., 1]
        elif np.sum(s_b[..., 2]) > 0:
            s_b = s_b[..., 2]
        else:
            s_b = s_b[..., 3]

        s_b = s_b.reshape([1] + list(exp_config.image_size[0:2]))
    elif exp_config.data_identifier == 'uzh_prostate':
        s_b = data.test.labels[index, ...]
        s_b = s_b[..., 0]
        s_b = s_b.reshape([1] + list(exp_config.image_size[0:2]))
    else:
        s_b = data.test.labels[index, ...].reshape([1] + list(exp_config.image_size[0:2]))



    x_b_for_cnt = utils.convert_to_uint8(np.squeeze(x_b.copy()))
    x_b_for_cnt = cv2.cvtColor(x_b_for_cnt, cv2.COLOR_GRAY2BGR)

    x_b_for_cnt = utils.resize_image(x_b_for_cnt, (2*ims[0], 2*ims[1]), interp=cv2.INTER_NEAREST)
    x_b_for_cnt = utils.histogram_equalization(x_b_for_cnt)

    for ss in range(3):

        print(ss)

        s_p_list = segvae_model.predict_segmentation_sample_levels(x_b, return_softmax=False)

        accum_list = [None]*exp_config.latent_levels
        accum_list[exp_config.latent_levels-1] = s_p_list[-1]
        for lvl in reversed(range(exp_config.latent_levels-1)):
            accum_list[lvl] = accum_list[lvl+1] + s_p_list[lvl]

        print('Plotting accum_list')
        for ii, img in enumerate(accum_list):

            plt.figure()
            img = utils.resize_image(np.squeeze(np.argmax(img, axis=-1)), (2*ims[0], 2*ims[1]), interp=cv2.INTER_NEAREST)
            plt.imshow(img[2*30:2*192-2*30,2*30:2*192-2*30], cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(outfolder, 'segm_lvl_%d_samp_%d.png' % (ii, ss)),bbox_inches='tight')

        print('Plotting s_p_list')
        for ii, img in enumerate(s_p_list):

            img = utils.softmax(img)

            plt.figure()
            img = utils.resize_image(np.squeeze(img[...,1]), (2*ims[0], 2*ims[1]), interp=cv2.INTER_NEAREST)
            plt.imshow(img[2*30:2*192-2*30,2*30:2*192-2*30], cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(outfolder, 'residual_lvl_%d_samp_%d.png' % (ii, ss)),bbox_inches='tight')

        s_p_d = np.uint8((np.squeeze(np.argmax(accum_list[0], axis=-1)) / (exp_config.nlabels-1)) * 255)
        s_p_d = utils.resize_image(s_p_d, (2*ims[0], 2*ims[1]), interp=cv2.INTER_NEAREST)

        print('Calculating contours')
        print(np.unique(s_p_d))
        rv = cv2.inRange(s_p_d, 84, 86)
        my = cv2.inRange(s_p_d, 169, 171)
        rv_cnt, hierarchy = cv2.findContours(rv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        my_cnt, hierarchy = cv2.findContours(my, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        x_b_for_cnt = cv2.drawContours(x_b_for_cnt, rv_cnt, -1, (0, 255, 0), 1)
        x_b_for_cnt = cv2.drawContours(x_b_for_cnt, my_cnt, -1, (0, 0, 255), 1)

    x_b_for_cnt = cv2.cvtColor(x_b_for_cnt, cv2.COLOR_BGR2RGB)

    print('Plotting final images...')
    plt.figure()
    plt.imshow(x_b_for_cnt[2*30:2*192-2*30,2*30:2*192-2*30,:], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(outfolder, 'input_img_cnts.png'),bbox_inches='tight')

    plt.figure()
    x_b = utils.convert_to_uint8(x_b)
    x_b = cv2.cvtColor(np.squeeze(x_b), cv2.COLOR_GRAY2BGR)
    x_b = utils.histogram_equalization(x_b)
    x_b = utils.resize_image(x_b, (2*ims[0], 2*ims[1]), interp=cv2.INTER_NEAREST)
    plt.imshow(x_b[2*30:2*192-2*30,2*30:2*192-2*30], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(outfolder, 'input_img.png'),bbox_inches='tight')

    plt.figure()
    s_b = utils.resize_image(np.squeeze(s_b), (2*ims[0], 2*ims[1]), interp=cv2.INTER_NEAREST)
    plt.imshow(s_b[2*30:2*192-2*30,2*30:2*192-2*30], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(outfolder, 'gt_seg.png'),bbox_inches='tight')


    # plt.show()


if __name__ == '__main__':

    base_path = sys_config.project_root

    # Code for selecting experiment from command line
    # parser = argparse.ArgumentParser(
    #     description="Script for a simple test loop evaluating a network on the test dataset")
    # parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    # args = parser.parse_args()


    # exp_path = args.EXP_PATH

    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/acdc/res128_probunet_exact_replication_bs32'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/acdc/rr_res128_hybrid_7_5_partlat_probunetopt_bs32'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/acdc/rr_res128_hybrid_7_5_partlat_probunetopt_bs32_logistic'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/acdc/res128_probunet_debugged'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/acdc/res128_hybrid_7_5_latentpart'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/acdc192/final_res192_hybrid_7_5_bs12'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/acdc192/final_res192_hybrid_7_5_bs12_partdep'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/acdc192/final_res128_probunet_bn_bs12'

    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/res128_hybrid_7_5_partlat_probunetopt_bs32_logistic_cont'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/final_res128_probunet_bn_bs12'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/final_res128_hybrid_7_5_bs12
    #
    exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/uzh_prostate/segvae_7_5_bs12_cont'

    model_path = os.path.join(base_path, exp_path)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config)
