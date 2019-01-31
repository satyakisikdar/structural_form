'''
MATLAB equivalent: scaledata.m
'''
import numpy as np

from simple_shift_scale import simple_shift_scale

def scale_data(data, params):
    '''
    scales the data according to several strategies
    :param data:
    :param params:
    :return:
    '''
    if params.runtime_type == 'relational':
        return data

    params.missing_data = np.sum(~np.isfinite(data), axis=0) > 0  # see if there's any NaNs or infs

    num_objects = data.shape[0]
    original_data = data
    data_mean = np.nanmean(data, axis=0)

    params.missing_data = np.sum(~np.isfinite(data), axis=0) > 0  # see if there's any NaNs or infs

    num_objects = data.shape[0]
    original_data = data
    data_mean = np.mean(data[np.isfinite(data)])  # double check the numbers to make sure it's matching with matlab
    data_stdev = np.std(data[np.isfinite(data)])

    params = make_chunks(data, params)

    if params.run_type == 'similarity':
        if params.similarity_transform == 'center':
            Z = np.eye(num_objects) - np.ones(num_objects) * (1 / num_objects)
            data = Z * data * Z

    elif params.run_type =='feature':
        if params.data_transform == 'simple_shift_scale':  # make data zero mean, covariance 1
            data = simple_shift_scale(data, params)

def make_chunks(data, params):
    '''
    TBD #TODO figure out what this does and if it is necessary
    :param data:
    :param params:
    :return:
    '''
    if params.missing_data:
        pass

    return params