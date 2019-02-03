'''
MATLAB files: setps.m, defaultps.m, setrunps.m
'''

import numpy as np
import os
import glob

import utilities

DATA_LOCATION = './matlab_codes/data/'

class Parameters:
    def __init__(self):
        with open('structures.txt') as struct_file, open('datasets.txt') as data_file:  # inserted an empty line to match MATLAB indices
            self.structures = list(structure.strip() for structure in struct_file)
            self.datasets = list(dataset.strip() for dataset in data_file)

        self.repeats = np.ones(len(self.datasets))
        self.data_locations = {dataset: f'{DATA_LOCATION}{dataset}.mat' for dataset in self.datasets}
        self.similarity_dimensions = {dataset: 1000 for dataset in self.datasets}

    def set_default_parameters(self):
        '''
        sets default parameter values
        :return:
        '''
        self.l_beta = 0.4  # expected branch length - parameter for exponential prior on branch-lengths
        self.sig_beta = 0.4  # expected value of 1/sigma  - parameter for exponential prior on 1/sigma
        self.sigma_init = 1 / self.sig_beta  # initial value for regularization parameter
        self.theta = None  # each additional node reduces prior by 3
        self.data_transform = None  # transformation unnecessary for relational data
        self.similarity_transform = None  # same

        ## ignore DISPLAY variables for now

        self.speed = 5  # 5: maximize approximate score (only optimize branch lengths as a heuristic
                        # if search is otherwise finished)

        self.fixed_all = 0  # all branch lengths must be identical
        self.fixed_internal = 0  # internal branch lengths must be identical
        self.fixed_external = 0  # external branch lengths must be identical
        self.product_tied = 0  # each branch length in a "direct product" graph (i.e. grid)
        # must equal the corresponding branch length from the corresponding component graph

        self.init = None  # options: None, ext, int, intext, fixedall

        # other parameters

        self.gibbs_clean = 1  # use heuristics to improve graph after each split
        self.outside_init = ''  # initialize with outside graph
        self.zgl_regularization = 0  # regularize using Zhu, Ghahramani and Lafferty: add terms
                                     # along the entire diagonal of the precision matrix

        self.edge_sum_steps = 10  # number of magnitudes
        self.edge_sum_lambda = 2  # base of magnitudes
        self.edge_offset = -5  # first magnitude is base^offset

        # initializing a relational structure:
        #   none:       initialize with all objects in one cluster
        #   overd:      use a structure created with one object per cluster
        #   external:   use a structure from self.relational_init_directory

        self.relational_outside_init = None
        self.relational_init_directory = ''  # use if relational_outside_init is not None


    def set_runtime_parameters(self, dataset_name):
        data_dictionary = utilities.load_from_mat(self.data_locations[dataset_name])
        data = data_dictionary['data']

        self.missing_data = False

        # data is always relational
        self.run_type = 'relational'
        self.num_objects = data['nobj']
        self.speed = 5
