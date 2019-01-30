'''
MATLAB files: setps.m, defaultps.m
'''

class Parameters:
    def __init__(self):
        self.structures = None # list of structures from setps.m  -- or have a dictionary with mappings like 'partition': 1, 'chain': 2, ...
        self.datasets = None # list of datasets from ...  -- or dictionary as above
        self.repeats = None  # numpy array of ones
        self.data_locations = None  # list of paths to datasets
        self.similarity_dimensions = None  # list/array of no of features

        ## add variables from setrunps.m


# data transform options
# DATA  -- which (if any) pre-processing steps to apply
# Feature data: simpleshiftscale -- make data zero mean, ensure that max
#				                    entry in the covariance matrix is 1
#		        makesimlike      -- transform data so that the largest
#				                    entry in the covariance matrix is 1, and the
#			                        smallest entry is 0
#	            none		 -- no pre-processing

# similarity matrix transform options
# Similarity data:  center	 -- center the similarity matrix
#	        	    none	 -- no pre-processing

class DefaultParameters(Parameters):
    def __init__(self):
        super().__init__()
        self.l_beta = 0.4  # expected branch length - parameter for exponential prior on branch-lengths
        self.sig_beta = 0.4  # expected value of 1/sigma  - parameter for exponential prior on 1/sigma
        self.sigma_init = 1 / self.sig_beta   # initial value for regularization parameter
        self.theta = None  # each additional node reduces prior by 3
        self.data_transform = None  # refer to above comment, make relevant data structures
        self.similarity_transform = None  # same

        ## ignore DISPLAY variables for now

        self.speed = None  #  4: only optimize branch lengths once per depth
                           # 5: maximize approximate score (only optimize branch lengths as a heuristic if search is otherwise finished)
                           # 54: speed 5 then 4

        ## follow the same style for the rest of the variables. insert _ between words
