'''
MATLAB file: runmodel.m
'''

def run_model(params, struct_index, data_index, repeat_index, save_file):
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
    log_probability = None
    graph = None
    names = None
    best_graph_log_probs = None
    best_graph = None

    return log_probability, graph, names, best_graph_log_probs, best_graph
