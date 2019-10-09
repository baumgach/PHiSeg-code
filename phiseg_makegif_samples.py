
import glob
import logging
import os
from importlib.machinery import SourceFileLoader

import cv2
import numpy as np

import config.system as sys_config
import utils
from data.data_switch import data_switch
from phiseg.phiseg_model import phiseg
# import scipy.misc
from PIL import Image


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


SAVE_VIDEO = False
SAVE_GIF = True
DISPLAY_VIDEO = True

video_target_size = (256, 256)


def histogram_equalization(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final


def main(model_path, exp_config):

    # Make and restore vagan model
    phiseg_model = phiseg(exp_config=exp_config)
    phiseg_model.load_weights(model_path, type='best_ged')

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    lat_lvls = exp_config.latent_levels

    # RANDOM IMAGE
    # x_b, s_b = data.test.next_batch(1)

    # FIXED IMAGE
    # Cardiac: 100 normal image
    # LIDC: 200 large lesion, 203, 1757 complicated lesion
    # Prostate: 165 nice slice, 170 is a challenging and interesting slice
    index = 165  # #

    if SAVE_GIF:
        outfolder_gif = os.path.join(model_path, 'model_samples_id%d_gif' % index)
        utils.makefolder(outfolder_gif)

    x_b = data.test.images[index,...].reshape([1]+list(exp_config.image_size))

    x_b_d = utils.convert_to_uint8(np.squeeze(x_b))
    x_b_d = utils.resize_image(x_b_d, video_target_size)

    if exp_config.data_identifier == 'uzh_prostate':
        # rotate
        rows, cols = x_b_d.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)
        x_b_d = cv2.warpAffine(x_b_d, M, (cols, rows))

    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outfile = os.path.join(model_path, 'model_samples_id%d.avi' % index)
        out = cv2.VideoWriter(outfile, fourcc, 5.0, (2*video_target_size[1], video_target_size[0]))

    samps = 20
    for ii in range(samps):

        # fix all below current level (the correct implementation)
        feed_dict = {}
        feed_dict[phiseg_model.training_pl] = False
        feed_dict[phiseg_model.x_inp] = x_b

        s_p, s_p_list = phiseg_model.sess.run([phiseg_model.s_out_eval, phiseg_model.s_out_eval_list], feed_dict=feed_dict)
        s_p = np.argmax(s_p, axis=-1)

        # s_p_d = utils.convert_to_uint8(np.squeeze(s_p))
        s_p_d = np.squeeze(np.uint8((s_p / exp_config.nlabels)*255))
        s_p_d = utils.resize_image(s_p_d, video_target_size, interp=cv2.INTER_NEAREST)

        if exp_config.data_identifier == 'uzh_prostate':
            #rotate
            s_p_d = cv2.warpAffine(s_p_d, M, (cols, rows))

        img = np.concatenate([x_b_d, s_p_d], axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = histogram_equalization(img)

        if exp_config.data_identifier == 'acdc':
            # labels (0 85 170 255)
            rv = cv2.inRange(s_p_d, 84, 86)
            my = cv2.inRange(s_p_d, 169, 171)
            rv_cnt, hierarchy = cv2.findContours(rv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            my_cnt, hierarchy = cv2.findContours(my, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(img, rv_cnt, -1, (0, 255, 0), 1)
            cv2.drawContours(img, my_cnt, -1, (0, 0, 255), 1)
        if exp_config.data_identifier == 'uzh_prostate':

            print(np.unique(s_p_d))
            s1 = cv2.inRange(s_p_d, 84, 86)
            s2 = cv2.inRange(s_p_d, 169, 171)
            # s3 = cv2.inRange(s_p_d, 190, 192)
            s1_cnt, hierarchy = cv2.findContours(s1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            s2_cnt, hierarchy = cv2.findContours(s2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # s3_cnt, hierarchy = cv2.findContours(s3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(img, s1_cnt, -1, (0, 255, 0), 1)
            cv2.drawContours(img, s2_cnt, -1, (0, 0, 255), 1)
            # cv2.drawContours(img, s3_cnt, -1, (255, 0, 255), 1)
        elif exp_config.data_identifier == 'lidc':
            thresh = cv2.inRange(s_p_d, 127, 255)
            lesion, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, lesion, -1, (0, 255, 0), 1)


        if SAVE_VIDEO:
            out.write(img)

        if SAVE_GIF:
            outfile_gif = os.path.join(outfolder_gif, 'frame_%s.png' % str(ii).zfill(3))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # scipy.misc.imsave(outfile_gif, img_rgb)
            im = Image.fromarray(img_rgb)
            im = im.resize((im.size[0]*2, im.size[1]*2), Image.ANTIALIAS)

            im.save(outfile_gif)

        if DISPLAY_VIDEO:
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if SAVE_VIDEO:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    base_path = sys_config.project_root

    # Code for selecting experiment from command line
    # parser = argparse.ArgumentParser(
    #     description="Script for a simple test loop evaluating a network on the test dataset")
    # parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    # args = parser.parse_args()


    # exp_path = args.EXP_PATH

    # exp_path = '/itet-stor/baumgach/net_scratch/logs/segvae/lidc/segvae_7_5'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/segvae/lidc/probunet'
    #
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/segvae/uzh_prostate_afterpaper/segvae_7_5_1annot'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/segvae/uzh_prostate_afterpaper/segvae_7_5'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/segvae/uzh_prostate_afterpaper/probUNET_1annotator_2'
    exp_path = '/itet-stor/baumgach/net_scratch/logs/segvae/uzh_prostate_afterpaper/segvae_7_5_batchnorm_rerun'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/segvae/uzh_prostate_afterpaper/segvae_7_5_batchnorm_schedule'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/segvae/uzh_prostate_afterpaper/probUNET'




    model_path = exp_path
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config)
