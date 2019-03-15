'''
MATLAB file: masterrun.m
'''
from parameters import Parameters

def main():
    master_file = None  # masterfile    = 'resultsdemo.mat';

    params = Parameters()
    params.set_default_parameters()

    ## ignore neato and other display functions for now

    params.rel_outside_init = 'overd'  # initialize relational structure with  one object per group

    use_structs = None  # structrures to use - eg: chain and ring - use the dictionary defined in Parameters class
    use_data = None  # datasets to use

    ## ignore extra data and structures for now

    struct_index_pair = None
    data_index_pair = None

    repeats = 1   # number of reapeats?

    for repeat_index in range(repeats):
        for ind in range(len(data_index_pair)):
            data_index = None
            struct_index = None
            print()  # print dataset and the structure
            # seed random gen with repeat_index

            pass

if __name__ == '__main__':
    main()
