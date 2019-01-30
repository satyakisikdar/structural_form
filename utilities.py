import scipy.io as sio
import numpy as np

def read_matlab_matrix(fname):
    '''

    :param fname: Filename to the MATLAB matrix - wrapper to the scipy io loadmat function
    :return: dictionary of names (rows), features (cols), data matrix - everything as a np array
    '''
    mat = sio.loadmat(fname)

    # flattening out the list of names and features into just strings
    flattened_names = []
    for name in mat['names']:
        flattened_names.append(name[0][0])
    mat['names'] = np.array(flattened_names)

    flattened_features = []
    for feature in mat['features']:
        flattened_features.append(feature[0][0])
    mat['features'] = np.array(flattened_features)

    # getting rid of the extra baggage
    mat.pop('__header__')
    mat.pop('__version__')
    mat.pop('__globals__')
    return mat