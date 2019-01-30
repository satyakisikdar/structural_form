'''
MATLAB files: setps.m, defaultps.m
'''

import math
import numpy as np
import os

DATA_LOCATION = './matlab_codes/data'

class Parameters:
    def __init__(self):
        with open('structures.txt', 'r') as structures:
            self.structures = set(structure.strip() for structure in structures)
        self.datasets = set(dataset.split('.')[0] for dataset in os.listdir(DATA_LOCATION))
        self.repeats = np.ones(len(self.datasets))
        self.data_locations = {dataset: data_location + dataset for dataset in self.datasets}
        self.similarity_dimensions = {dataset: 1000 for dataset in self.datasets}

    def runtime_parameters(self, dataset):
        # TODO: update function call for mat -> dict conversion
        data = load_from_mat(self.data_locations[dataset])

        if 'type' in data:
            self.run_type = 'relational'
            self.num_objects = data[nobj]
            self.speed = 5
        elif data.size[0] == data.size[1] and not self.feature_force:
            self.run_type = 'similarity'
            self.num_objects = data.size[0]
            self.run_dimension = self.similarity_dimensions[dataset]
        else:
            self.run_type = 'feature'
            self.num_objects = data.size[0]


# data transform options
# DATA  -- which (if any) pre-processing steps to apply
# Feature data: simpleshiftscale -- make data zero mean, ensure that max entry in the covariance matrix is 1
#		   makesimlike   -- transform data so that the largest entry in the covariance matrix is 1,
#                                   and the smallest entry is 0
#	            none         -- no pre-processing

# similarity matrix transform options
# Similarity data:  center	 -- center the similarity matrix
#	    	    none	 -- no pre-processing

class DefaultParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.l_beta = 0.4  # expected branch length - parameter for exponential prior on branch-lengths
        self.sig_beta = 0.4  # expected value of 1/sigma  - parameter for exponential prior on 1/sigma
        self.sigma_init = 1 / self.sig_beta     # initial value for regularization parameter
        self.theta = None                       # each additional node reduces prior by 3
        self.data_transform = None              # refer to above comment, make relevant data structures
        self.similarity_transform = None        # same

        ## ignore DISPLAY variables for now

        self.speed = None           #  4: only optimize branch lengths once per depth
                                    # 5: maximize approximate score (only optimize branch lengths
                                    # as a heuristic if search is otherwise finished)
                                    # 54: speed 5 then 4

        self.fixed_all      = 0     # all branch lengths must be identical
        self.fixed_internal = 0     # internal branch lengths must be identical
        self.fixed_external = 0     # external branch lengths must be identical
        self.product_tied   = 0     # each branch length in a "direct product" graph (i.e. grid)
                                    # must equal the corresponding branch length from the
                                    # corresponding component graph

        self.init = None            # options: None, ext, int, intext, fixedall

        # other parameters

        self.gibbs_clean = 1        # use heuristics to improve graph after each split
        self.outside_init = ''      # initialize with outside graph
        self.zgl_regularization = 0 # regularize using Zhu, Ghahramani and Lafferty: add terms
                                    # along the entire diagonal of the precision matrix
        self.feature_force = 1      # set flag to analyze as a square feature matrix, not a
                                    # similarity matrix (which is default)

        self.edge_sum_steps = 10    # number of magnitudes
        self.edge_sum_lambda = 2    # base of magnitudes
        self.edge_offset = -5       # first magnitude is base^offset

        # initializing a relational structure:
        #   none:       initialize with all objects in one cluster
        #   overd:      use a structure created with one object per cluster
        #   external:   use a structure from self.relational_init_directory

        self.relational_outside_init = None
        self.relational_init_directory = '' # use if relational_outside_init is not None
