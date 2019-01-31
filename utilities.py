import scipy.io as sio
import numpy as np

def load_from_mat(fname):
    '''

    :param fname: Filename to the MATLAB matrix - wrapper to the scipy io loadmat function
    :return: dictionary of fields and values. every value is a numpy array.
    '''
    mat = sio.loadmat(fname)

    # flattening out attributes other than matrices
    new_mat = {}
    for attr_name, attrs in mat.items():
        if attr_name.startswith('__'):  # getting rid of the meta variables
            continue
        if len(attrs.shape) > 1 and attrs.shape[1] > 1:  # if the data is 2D, with more than 1 col, keep as is
            new_mat[attr_name] = attrs
            continue
        new_mat[attr_name] = np.array(attrs[0])  # flatten this attribute

    return new_mat