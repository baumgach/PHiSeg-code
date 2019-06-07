
def data_switch(data_identifier):

    if data_identifier == 'acdc':
        from data.acdc_data import acdc_data as data_loader
    elif data_identifier == 'lidc':
        from data.lidc_data import lidc_data as data_loader
    elif data_identifier == 'uzh_prostate':
        from data.uzh_prostate_data import uzh_prostate_data as data_loader
    else:
        raise ValueError('Unknown data identifier: %s' % data_identifier)

    return data_loader