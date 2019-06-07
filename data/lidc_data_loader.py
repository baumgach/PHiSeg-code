# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import numpy as np
import logging
import h5py
import pickle
from sklearn.model_selection import train_test_split

import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def find_subset_for_id(ids_dict, id):

    for tt in ['test', 'train', 'val']:
        if id in ids_dict[tt]:
            return tt
    raise ValueError('id was not found in any of the train/test/val subsets.')


def prepare_data(input_file, output_file):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    hdf5_file = h5py.File(output_file, "w")
    max_bytes = 2 ** 31 - 1

    data = {}
    file_path = os.fsdecode(input_file)
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    new_data = pickle.loads(bytes_in)
    data.update(new_data)

    series_uid = []

    for key, value in data.items():
        series_uid.append(value['series_uid'])

    unique_subjects = np.unique(series_uid)

    split_ids = {}
    train_and_val_ids, split_ids['test'] = train_test_split(unique_subjects, test_size=0.2)
    split_ids['train'], split_ids['val'] = train_test_split(train_and_val_ids, test_size=0.2)

    images = {}
    labels = {}
    uids = {}
    groups = {}

    for tt in ['train', 'test', 'val']:
        images[tt] = []
        labels[tt] = []
        uids[tt] = []
        groups[tt] = hdf5_file.create_group(tt)

    for key, value in data.items():

        s_id = value['series_uid']

        tt = find_subset_for_id(split_ids, s_id)

        images[tt].append(value['image'].astype(float)-0.5)

        lbl = np.asarray(value['masks'])  # this will be of shape 4 x 128 x 128
        lbl = lbl.transpose((1,2,0))

        labels[tt].append(lbl)
        uids[tt].append(hash(s_id))  # Checked manually that there are no collisions

    for tt in ['test', 'train', 'val']:

        groups[tt].create_dataset('uids', data=np.asarray(uids[tt], dtype=np.int))
        groups[tt].create_dataset('labels', data=np.asarray(labels[tt], dtype=np.uint8))
        groups[tt].create_dataset('images', data=np.asarray(images[tt], dtype=np.float))

    hdf5_file.close()


def load_and_maybe_process_data(input_file,
                                preprocessing_folder,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the LIDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    data_file_name = 'data_lidc.hdf5'

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_file, data_file_path)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    input_file = '/itet-stor/baumgach/bmicdatasets-originals/Originals/LIDC-IDRI/data_lidc.pickle'
    preprocessing_folder = '/srv/glusterfs/baumgach/preproc_data/lidc'

    d = load_and_maybe_process_data(input_file, preprocessing_folder, force_overwrite=True)

