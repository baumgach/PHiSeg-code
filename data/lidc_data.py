# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import numpy as np
from data import lidc_data_loader
from data.batch_provider import BatchProvider

class lidc_data():

    def __init__(self, exp_config):

        data = lidc_data_loader.load_and_maybe_process_data(
            input_file=exp_config.data_root,
            preprocessing_folder=exp_config.preproc_folder,
            force_overwrite=False,
        )

        self.data = data

        # Extract the number of training and testing points
        indices = {}
        for tt in data:
            N = data[tt]['images'].shape[0]
            indices[tt] = np.arange(N)

        # Create the batch providers
        augmentation_options = exp_config.augmentation_options

        # Backwards compatibility, TODO remove for final version
        if not hasattr(exp_config, 'annotator_range'):
            exp_config.annotator_range = range(exp_config.num_labels_per_subject)

        self.train = BatchProvider(data['train']['images'], data['train']['labels'], indices['train'],
                                   add_dummy_dimension=True,
                                   do_augmentations=True,
                                   augmentation_options=augmentation_options,
                                   num_labels_per_subject=exp_config.num_labels_per_subject,
                                   annotator_range=exp_config.annotator_range)
        self.validation = BatchProvider(data['val']['images'], data['val']['labels'], indices['val'],
                                        add_dummy_dimension=True,
                                        num_labels_per_subject=exp_config.num_labels_per_subject,
                                        annotator_range=exp_config.annotator_range)
        self.test = BatchProvider(data['test']['images'], data['test']['labels'], indices['test'],
                                  add_dummy_dimension=True,
                                  num_labels_per_subject=exp_config.num_labels_per_subject,
                                  annotator_range=exp_config.annotator_range)

        self.test.images = data['test']['images']
        self.test.labels = data['test']['labels']


if __name__ == '__main__':

    # If the program is called as main, perform some debugging operations
    from phiseg.experiments import segvae_lidc_res128_phiseg as exp_config
    data = lidc_data(exp_config)

    print(data.data['val']['images'].shape[0])
    print(data.data['test']['images'].shape[0])
    print(data.data['train']['images'].shape[0])
    print(data.data['train']['images'].shape[0]+data.data['test']['images'].shape[0]+data.data['val']['images'].shape[0])

    print('DEBUGGING OUTPUT')
    print('training')
    for ii in range(2):
        X_tr, Y_tr = data.train.next_batch(10)
        print(np.mean(X_tr))
        print(Y_tr.shape)
        print('--')

    print('test')
    for ii in range(2):
        X_te, Y_te = data.test.next_batch(10)
        print(np.mean(X_te))
        print(Y_te.shape)
        print('--')

    print('validation')
    for ii in range(2):
        X_va, Y_va = data.validation.next_batch(10)
        print(np.mean(X_va))
        print(Y_va.shape)
        print('--')
