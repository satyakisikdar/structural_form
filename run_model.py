'''
MATLAB file: runmodel.m
'''
import os
import numpy as np

from utilities import load_from_mat
from struct_counts import struct_counts
from structure_fit import structure_fit
from StructGraph import StructGraph

def rel_graph_init(data, z, params):
    '''

    :param data:
    :param z: dictionary of cluster assignments
    :param params:
    :return: graph
    '''

    struct_graph = StructGraph(struct_type=params.run_struct_name, obj_count=params.num_objects, sigma=params.sigma,
                        cluster_labels=z)

    if params.run_struct_name in ('undirchain', 'undirring', 'undirhierarchy', 'undirchainnoself',
                                  'undirringnoself', 'undirhierarchynoself'):
        struct_graph.to_undirected()  # make it undirected

    return struct_graph


def branch_length_cases(data, params, graph, best_graph_log_probs, best_graph, save_file):
    '''
    deal with different approaches to branchlengths at current speed
    :param data:
    :param params:
    :param graph:
    :param best_graph_log_probs:
    :param best_graph:
    :param save_file:
    :return:
    log_likelihood:
    graph:
    best_graph_log_probs:
    best_graph:
    params:
    '''
    if params.init == 'none':
        log_likelihood, graph, best_graph_log_probs, best_graph = structure_fit(data, params, graph, f'{save_file}_noinit_{params.speed}')

    elif params.init == 'ext':
        params.fixed_external = True
        log_likelihood, graph, best_graph_log_probs, best_graph = structure_fit(data, params, graph, f'{save_file}_exttie_{params.speed}')
        print('untie external...')

        params.fixed_external = False


def run_model(params, struct_index, data_index, save_file):
    '''
    Given data set DIND, find the best instance of form SIND.
    :param params: DefaultParameters object
    :param struct_index: index to structure
    :param data_index: index to dataset
    :param repeat_index: which repeat is this?
    :param save_file: where to save interim results
    :return:
        log_probability: log probability of the best structure found
        graph: the best substructure found
        names: TBD
        best_graph_log_probs: log probabilities of structures explored along the way
        best_graph: structures along the way
    '''
    # log_probability = None
    # graph = None
    # names = None
    # best_graph_log_probs = None
    # best_graph = None

    struct_name = params.structures[struct_index]
    params.run_struct_name =  struct_name # equiv of ps.runps.structname

    file_root = f'./results/{params.structures[struct_index]}/out/{params.datasets[data_index]}'
    if not os.path.exists(file_root):
        os.makedirs(file_root)

    mat = load_from_mat(params.data_locations[params.datasets[data_index]])
    params.set_runtime_parameters(dataset_name=struct_name)

    names = [str(i) for i in range(params.num_objects)]

    if params.relational_outside_init == 'overd':
        if struct_name in ('dirchain', 'dirchainnoself', 'dirring', 'dirringnoself', 'dirhierarchy',
                           'dirhierarchynoself','undirchain', 'undirchainnoself', 'undirring', 'undirringnoself',
	                       'undirhierarchy', 'undirhierarchynoself'):

            best_z = {obj: obj for obj in range(params.num_objects)}  # initially each object is in its own cluster
            graph = rel_graph_init(data=mat['data'], z=best_z, params=params)

    params =  struct_counts(params, params.num_objects)

    log_likelihood, graph, best_graph_log_probs, best_graph, params = branch_length_cases(data, params, graph, best_graph_log_probs,
                                                                                 best_graph, save_file)

    return log_likelihood, graph, names, best_graph_log_probs, best_graph
