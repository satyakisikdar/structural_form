'''
simple_shift_scale.m
'''
import numpy as np

def simple_shift_scale(data, params):
    '''
    shift and scale data so that mean is zero, and largest covariance is 1
    :param data:
    :param params:
    :return:
    '''

    original_data = data

    global_mean = None # TODO: check this out
    data = data - global_mean

    if params.missing_data:
        pass

    return data