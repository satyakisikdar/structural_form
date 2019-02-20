#!/usr/bin/env python3

import networkx as nx
import numpy as np

from scipy.io import loadmat, savemat

def graph_to_matlab(graph, directory='./matlab_codes/data', data_type='relbin'):
    ''' converts networkx graph to .mat file to use in K&T code
        @param name: name of dataset to import
        @param directory: data directory (default ./matlab_codes/data)
        @param data_type: specify type of data to be read in by K&T (default relbin)
    '''
    adj_matrix = nx.to_numpy_matrix(graph)
    struct = {'data': {'R': adj_matrix, 'type': data_type, 'nobj': adj_matrix.shape[0]}}
    savemat('{}/{}.mat'.format(directory, graph.name), struct)


def matlab_to_graph(name, directory='./matlab_codes/data'):
    ''' converts .mat file containing relational data from K&T format to networkx graph
        @param name: name of dataset to import
        @param directory: data directory (default ./matlab_codes/data)
        @return networkx directed graph
    '''
    mat = loadmat('{}/{}.mat'.format(directory.strip('/'), name))
    matrix, data_type, size = mat['data'][0][0]
    return nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())


def import_results(dataset, form, directory='./matlab_codes/results', repeat=1):
    ''' imports list of result history from matlab
        @param dataset:     name of dataset to import
        @param form:        name of basic form used
        @param directory:   results directory (default ./matlab_codes/results)
        @param repeat:      repeat number for result to import (always 1 for relbin data)
        @return list of dictionaries containing log likelihood score and z array for each split
    '''
    directory = directory.strip('/')
    mat = loadmat('{}/{}out/{}{}/growthhistorynoinit5.mat'.format(directory, form, dataset, repeat))

    graph_history = [{'log_likelihood': log_lik} for log_lik in mat['bestgraphlls'][0]]

    for count, graph in enumerate(mat['bestgraph'][0]):
        graph_history[count]['z'] = graph[0][0][8][0] # indexing to get graph.z

    return graph_history
    

if __name__ == '__main__':
    g = matlab_to_graph('prisoners', './matlab_codes/data')

    g.name = 'prisoners2'
    graph_to_matlab(g, './matlab_codes/data')

    print(import_results('prisoners', 'chain'))
