'''
MATLAB files: setps.m, defaultps.m, setrunps.m
'''

import numpy as np
import os
import glob

import utilities
from graph_priors import set_log_priors

DATA_LOCATION = './matlab_codes/data/'

class Parameters:
    def __init__(self):
        with open('structures.txt') as struct_file, open('datasets.txt') as data_file:  # inserted an empty line to match MATLAB indices
            self.structures = list(structure.strip() for structure in struct_file)
            self.datasets = list(dataset.strip() for dataset in data_file)

        # self.repeats = np.ones(len(self.datasets))
        self.data_locations = {dataset: f'{DATA_LOCATION}{dataset}.mat' for dataset in self.datasets}
        # self.similarity_dimensions = {dataset: 1000 for dataset in self.datasets}
        self.set_default_parameters()

    def set_default_parameters(self):
        '''
        sets default parameter values
        :return:
        '''
        self.l_beta = 0.4  # expected branch length - parameter for exponential prior on branch-lengths
        self.sig_beta = 0.4  # expected value of 1/sigma  - parameter for exponential prior on 1/sigma
        self.sigma = 1 / self.sig_beta  # initial value for regularization parameter
        self.theta = 1 - np.exp(-3)  # each additional node reduces prior by 3
        self.data_transform = None  # transformation unnecessary for relational data
        self.similarity_transform = None  # same

        ## ignore DISPLAY variables for now

        self.speed = 5  # 5: maximize approximate score (only optimize branch lengths as a heuristic
                        # if search is otherwise finished)

        self.fixed_all = False  # all branch lengths must be identical
        self.fixed_internal = False  # internal branch lengths must be identical
        self.fixed_external = False  # external branch lengths must be identical
        self.product_tied = False  # each branch length in a "direct product" graph (i.e. grid)
        # must equal the corresponding branch length from the corresponding component graph

        self.init = None  # options: None, ext, int, intext, fixedall

        # other parameters

        self.gibbs_clean = True  # use heuristics to improve graph after each split
        # self.outside_init = ''  # initialize with outside graph - we dont care about this
        self.zgl_regularization = False # regularize using Zhu, Ghahramani and Lafferty: add terms
                                     # along the entire diagonal of the precision matrix

        self.edge_sum_steps = 10  # number of magnitudes
        self.edge_sum_lambda = 2  # base of magnitudes
        self.edge_offset = -5  # first magnitude is base^offset

        # initializing a relational structure:
        #   none:       initialize with all objects in one cluster
        #   overd:      use a structure created with one object per cluster
        #   external:   use a structure from self.relational_init_directory

        # self.relational_outside_init = None
        # self.relational_init_directory = ''  # use if relational_outside_init is not None


    def set_runtime_parameters(self, data_graph, struct_name):
        '''
        sets runtime params including log_priors
        :param data_graph:
        :return:
        '''
        self.missing_data = False
        self.struct_name = struct_name
        # data is always relational
        self.run_type = 'relational'
        self.num_objects = data_graph.order()
        self.speed = 5

        set_log_priors(self, num_objects=self.num_objects) # log priors of structures - values are computed in struct_counts
