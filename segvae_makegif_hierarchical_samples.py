
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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


SAVE_VIDEO = True

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
    segvae_model = segvae(exp_config=exp_config)
    segvae_model.load_weights(model_path, type='best_ged')

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    lat_lvls = exp_config.latent_levels

    # RANDOM IMAGE
    # x_b, s_b = data.test.next_batch(1)

    # FIXED IMAGE
    # Cardiac: 100 normal image
    # LIDC: 200 large lesion, 203, 1757 complicated lesion
    # Prostate: 165 nice slice
    index = 165 #

    x_b = data.test.images[index,...].reshape([1]+list(exp_config.image_size))
    if exp_config.data_identifier == 'lidc':
        s_b = data.test.labels[index,...]
        if np.sum(s_b[...,0]) > 0:
            s_b = s_b[...,0]
        elif np.sum(s_b[...,1]) > 0:
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
    #
    # print(x_b.shape)
    # print(s_b.shape)

    # x_b[:,30:64+10,64:64+10,:] = np.mean(x_b)
    #
    # x_b = utils.add_motion_artefacts(np.squeeze(x_b), 15)
    # x_b = x_b.reshape([1]+list(exp_config.image_size))

    x_b_d = utils.convert_to_uint8(np.squeeze(x_b))
    x_b_d = utils.resize_image(x_b_d, video_target_size)

    s_b_d = np.squeeze(np.uint8((s_b / exp_config.nlabels)*255))
    s_b_d = utils.resize_image(s_b_d, video_target_size, interp=cv2.INTER_NEAREST)

    _, mu_list_init, _ = segvae_model.generate_prior_samples(x_b, return_params=True)

    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outfile = os.path.join(model_path, 'samplevid_id%d.avi' % index)
        out = cv2.VideoWriter(outfile, fourcc, 10.0, (3*video_target_size[1], video_target_size[0]))

    for lvl in reversed(range(lat_lvls)):

        samps = 50 if lat_lvls > 1 else 200
        for _ in range(samps):

            # z_list, mu_list, sigma_list = segvae_model.generate_prior_samples(x_b, return_params=True)

            print('doing level %d/%d' % (lvl, lat_lvls))

            # fix all below current level
            # for jj in range(lvl,lat_lvls-1):
            #     z_list[jj+1] = mu_list_init[jj+1]  # fix jj's level to mu



            # sample only current level
            # z_list_new = z_list.copy()
            # for jj in range(lat_lvls):
            #     z_list_new[jj] = mu_list_init[jj]
            # z_list_new[lvl] = z_list[lvl]
            # z_list = z_list_new
            #
            # print('z means')
            # for jj, z in enumerate(z_list):
            #     print('lvl %d: %.3f' % (jj, np.mean(z)))
            #
            #
            # feed_dict = {i: d for i, d in zip(segvae_model.prior_z_list_gen, z_list)}
            # feed_dict[segvae_model.training_pl] = False
            #

            # fix all below current level (the correct implementation)
            feed_dict = {}
            for jj in range(lvl,lat_lvls-1):
                feed_dict[segvae_model.prior_z_list_gen[jj+1]] = mu_list_init[jj+1]
            feed_dict[segvae_model.training_pl] = False
            feed_dict[segvae_model.x_inp] = x_b

            s_p, s_p_list = segvae_model.sess.run([segvae_model.s_out_eval, segvae_model.s_out_eval_list], feed_dict=feed_dict)
            s_p = np.argmax(s_p, axis=-1)

            print(np.unique(s_p))

            # print('mean logits for myo cardium per level')
            # fig = plt.figure()
            #
            # cumsum = np.zeros((128,128))
            # cumsum_all = np.zeros((128,128,4))
            # for i, s in enumerate(reversed(s_p_list)):
            #
            #     cumsum += s[0,:,:,2]
            #     cumsum_all += s[0,:,:,:]
            #
            #     fig.add_subplot(4,4,i+1)
            #     plt.imshow(s[0,:,:,2])
            #
            #     fig.add_subplot(4,4,i+1+4)
            #     plt.imshow(cumsum)
            #
            #     fig.add_subplot(4,4,i+1+8)
            #     plt.imshow(1./(1+np.exp(-cumsum)))
            #
            #     fig.add_subplot(4,4,i+1+12)
            #     plt.imshow(np.argmax(cumsum_all, axis=-1))
            #
            #
            # plt.show()



            # DEUBG
            # cum_img = np.squeeze(s_p_list[lat_lvls-1])
            # cum_img_disp = softmax(cum_img)
            #
            # indiv_img = np.squeeze(s_p_list[lat_lvls-1])
            # indiv_img_disp = softmax(indiv_img)
            #
            # for ii in reversed(range(lat_lvls-1)):
            #     cum_img += np.squeeze(s_p_list[ii])
            #     indiv_img = np.squeeze(s_p_list[ii])
            #
            #     cum_img_disp = np.concatenate([cum_img_disp, softmax(cum_img)], axis=1)
            #     indiv_img_disp = np.concatenate([indiv_img_disp, softmax(indiv_img)], axis=1)
            #
            #
            # cum_img_disp = utils.convert_to_uint8(np.argmax(cum_img_disp, axis=-1))
            # indiv_img_disp = utils.convert_to_uint8(indiv_img_disp[:,:,2])
            #
            # cum_img_disp = np.concatenate([cum_img_disp, indiv_img_disp], axis=0)
            #
            #
            # print('cum img shape')
            # print(cum_img_disp.shape)
            # cv2.imshow('debug', cum_img_disp)
            # END DEBUG

            # s_p_d = utils.convert_to_uint8(np.squeeze(s_p))
            s_p_d = np.squeeze(np.uint8((s_p / exp_config.nlabels)*255))
            s_p_d = utils.resize_image(s_p_d, video_target_size, interp=cv2.INTER_NEAREST)

            img = np.concatenate([x_b_d, s_b_d, s_p_d], axis=1)
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
                # labels (0 85 170 255)
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

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Sampling level %d/%d' % (lvl+1, lat_lvls), (30, 256-30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            print('actual size')
            print(img.shape)

            if SAVE_VIDEO:
                out.write(img)

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

    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/segvae_7_5'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/lidc/probunet'
    #
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/uzh_prostate_afterpaper/segvae_7_5_1annot'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/uzh_prostate_afterpaper/segvae_7_5'
    # exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/uzh_prostate_afterpaper/probUNET_1annotator_2'
    exp_path = '/itet-stor/baumgach/net_scratch/logs/phiseg/uzh_prostate_afterpaper/segvae_7_5_batchnorm_rerun'




    model_path = exp_path
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config)
