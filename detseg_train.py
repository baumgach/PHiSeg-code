# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

from data.data_switch import data_switch
from detseg.model_segmenter import segmenter

def main():

    # Select experiment below
    # from detseg.experiments import acdc_unet as exp_config
    from detseg.experiments import acdc_probunetarch as exp_config
    # from detseg.experiments import lidc_probunetarch as exp_config
    # from detseg.experiments import uzh_prostate_probunetarch as exp_config
    # from detseg.experiments import twolbl_uzh_prostate_probunetarch as exp_config

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    # Build VAGAN model
    segmenter_model = segmenter(exp_config=exp_config, data=data, fixed_batch_size=exp_config.batch_size)

    # Train VAGAN model
    segmenter_model.train()


if __name__ == '__main__':

    main()
