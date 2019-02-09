'''
Structure Graph container
'''
import networkx as nx

class StructGraph():
    '''
    g.objcount:	        number of objects in the structure

    g.adjcluster:	    a cluster graph represented as an adjacency matrix (a
                        cluster graph is a graph where the nodes correspond to
                        clusters, not individual objects)

    g.adjclustersym:    symmetric version of g.adjcluster

    g.Wcluster:	        a weighted cluster graph. The weight of an edge is the
                        reciprocal of its length

    g.adj:		        a graph including object nodes and cluster nodes. The
                        first g.objcount nodes are object nodes, and the rest
                        are cluster nodes.

    g.W:		        a weighted version of g.adj

    g.z:		        cluster assignments for the objects

    g.ncomp:	        number of graph components. Will be 2 for direct
                        products (e.g. a cylinder has two components -- a chain and a ring)

    g.components:	    cell array storing the graph components

    g.extlen:	        length of all external branches (branches that connect an
                        object to a cluster). Used only if all external branch
                        lengths are tied.

    g.intlen:	        length of internal branches (branches that connect a
                        cluster to a cluster. Used only if all internal branch
                        lengths are tied.
    '''
    def __init__(self, struct_type, obj_count, sigma, cluster_labels, external_branch_len=1, internal_branch_len=1):
        self.struct_type = struct_type  # type of structure -
        self.graph = nx.DiGraph()  # the graph container
        self.obj_count = obj_count  # number of objects in the structure - number of clusters
        self.sigma = sigma
        self.cluster_labels = cluster_labels
        self.external_branch_len = external_branch_len
        self.internal_branch_len = internal_branch_len

    def to_undirected(self):
        '''
        turns the graph container to undirected
        :return:
        '''
        self.graph.to_undirected()