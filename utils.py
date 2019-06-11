# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import nibabel as nib
import numpy as np
import os
import logging
from skimage import measure, transform
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from medpy.metric import jc

try:
    import cv2
except:
    logging.warning('Could not import opencv. Augmentation functions will be unavailable.')
else:
    def rotate_image(img, angle, interp=cv2.INTER_LINEAR):

        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)

    def rotate_image_as_onehot(img, angle, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = rotate_image(convert_to_onehot(img, nlabels=nlabels), angle, interp)
        return np.argmax(onehot_output, axis=-1)

    def resize_image(im, size, interp=cv2.INTER_LINEAR):

        im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
        return im_resized

    def resize_image_as_onehot(im, size, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = resize_image(convert_to_onehot(im, nlabels), size, interp=interp)
        return np.argmax(onehot_output, axis=-1)


    def deformation_to_transformation(dx, dy):

        nx, ny = dx.shape

        # grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
        grid_y, grid_x = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")  # Robin's change to make it work with non-square images

        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        return map_x, map_y

    def dense_image_warp(im, dx, dy, interp=cv2.INTER_LINEAR, do_optimisation=True):

        map_x, map_y = deformation_to_transformation(dx, dy)

        # The following command converts the maps to compact fixed point representation
        # this leads to a ~20% increase in speed but could lead to accuracy losses
        # Can be uncommented
        if do_optimisation:
            map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)
        return cv2.remap(im, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT) #borderValue=float(np.min(im)))


    def dense_image_warp_as_onehot(im, dx, dy, nlabels, interp=cv2.INTER_LINEAR, do_optimisation=True):

        onehot_output = dense_image_warp(convert_to_onehot(im, nlabels), dx, dy, interp, do_optimisation=do_optimisation)
        return np.argmax(onehot_output, axis=-1)


def find_floor_in_list(l, t):
    # Linear, because not important enough to optimize

    max_smallest = -np.inf
    argmax_smallest = None

    for i, n in enumerate(l):
        if t >= n and n > max_smallest:
            max_smallest = n
            argmax_smallest = i

    if argmax_smallest is None:
        raise ValueError("All elements in list l are larger than t=%d" % t)

    return max_smallest, argmax_smallest

def convert_to_onehot(lblmap, nlabels):

    output = np.zeros((lblmap.shape[0], lblmap.shape[1], nlabels))
    for ii in range(nlabels):
        output[:,:,ii] = (lblmap == ii).astype(np.uint8)
    return output

def convert_batch_to_onehot(lblbatch, nlabels):

    out = []
    for ii in range(lblbatch.shape[0]):

        lbl = convert_to_onehot(lblbatch[ii,...], nlabels)
        out.append(lbl)

    return np.asarray(out)

def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a,v)


def norm_l2(a,v):

    a = a.flatten()
    v = v.flatten()

    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)

    return np.mean(np.sqrt(a**2 + v**2))



def all_argmax(arr, axis=None):

    return np.argwhere(arr == np.amax(arr, axis=axis))


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def create_and_save_nii(data, img_path):

    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, img_path)



class Bunch:
    # Useful shortcut for making struct like contructs
    # Example:
    # mystruct = Bunch(a=1, b=2)
    # print(mystruct.a)
    # >>> 1
    def __init__(self, **kwds):
        self.__dict__.update(kwds)



def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)
#

def convert_to_uint8_rgb_fixed(image):
    image = (image + 1) * 127.5
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def map_image_to_intensity_range(image, min_o, max_o, percentiles=0):

    # If percentile = 0 uses min and max. Percentile >0 makes normalisation more robust to outliers.

    if image.dtype in [np.uint8, np.uint16, np.uint32]:
        assert min_o >= 0, 'Input image type is uintXX but you selected a negative min_o: %f' % min_o

    if image.dtype == np.uint8:
        assert max_o <= 255, 'Input image type is uint8 but you selected a max_o > 255: %f' % max_o

    min_i = np.percentile(image, 0 + percentiles)
    max_i = np.percentile(image, 100 - percentiles)

    image = (np.divide((image - min_i), max_i - min_i) * (max_o - min_o) + min_o).copy()

    image[image > max_o] = max_o
    image[image < min_o] = min_o

    return image


def map_images_to_intensity_range(X, min_o, max_o, percentiles=0):

    X_mapped = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_mapped[ii,...] = map_image_to_intensity_range(Xc, min_o, max_o, percentiles)

    return X_mapped.astype(np.float32)


def normalise_images(X):
    '''
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    '''

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_white[ii,...] = normalise_image(Xc)

    return X_white.astype(np.float32)

def jaccard_onehot(pred, gt):

    # assuming last dimension is classes

    intersection = np.sum(pred*gt)
    pred_count = np.sum(pred)
    gt_count = np.sum(gt)

    # FN = np.sum((1-pred)*gt)
    # FP = np.sum(pred*(1-gt))
    #
    # return TP / (TP + FN + FP)

    return intersection / (pred_count + gt_count - intersection)


def generalised_energy_distance(sample_arr, gt_arr, nlabels, **kwargs):

    def dist_fct(m1, m2):

        label_range = kwargs.get('label_range', range(nlabels))

        per_label_iou = []
        for lbl in label_range:

            # assert not lbl == 0  # tmp check
            m1_bin = (m1 == lbl)*1
            m2_bin = (m2 == lbl)*1

            if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
                per_label_iou.append(1)
            elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
                per_label_iou.append(0)
            else:
                per_label_iou.append(jc(m1_bin, m2_bin))

        # print(1-(sum(per_label_iou) / nlabels))

        return 1-(sum(per_label_iou) / nlabels)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    for i in range(N):
        for j in range(M):
            # print(dist_fct(sample_arr[i,...], gt_arr[j,...]))
            d_sy.append(dist_fct(sample_arr[i,...], gt_arr[j,...]))

    for i in range(N):
        for j in range(N):
            # print(dist_fct(sample_arr[i,...], sample_arr[j,...]))
            d_ss.append(dist_fct(sample_arr[i,...], sample_arr[j,...]))

    for i in range(M):
        for j in range(M):
            # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
            d_yy.append(dist_fct(gt_arr[i,...], gt_arr[j,...]))

    return (2./(N*M))*sum(d_sy) - (1./N**2)*sum(d_ss) - (1./M**2)*sum(d_yy)


# import matplotlib.pyplot as plt
def variance_ncc_dist(sample_arr, gt_arr):

    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):


        log_samples = np.log(m_samp + eps)

        return -1.0*np.sum(m_gt*log_samples, axis=-1)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(ncc_list)


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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def list_mean(lst):

    N = len(lst)
    return (1./N)*sum(lst)