import networkx as nx
import random

from typing import List, Tuple, Set, Any
from copy import deepcopy
from itertools import permutations, combinations

from graph_likelihoods import graph_likelihood
from graph_priors import graph_prior
from parameters import Parameters


class ClusterGraph(nx.DiGraph):
    '''
    Structure for Cluster Graph
    Analogous to Graph in MATLAB
    '''
    def __init__(self, struct_type: str, dataset: str, data_graph: nx.DiGraph, sigma=None) -> None:
        nx.DiGraph.__init__(self)
        self.data_graph = data_graph
        self.struct_type = struct_type
        self.dataset = dataset
        self.sigma = sigma
        self.number_empty_nodes = 0  # empty nodes for the tree
        self.num_objects = 0
        # self.params = params
        # self.params.set_runtime_parameters(data_graph=data_graph, struct_name=self.struct_type)
        self.cluster_labels = dict()   # cluster assignments, maps nodes to clusters

        self.add_node(0, members=set(data_graph.nodes()))  # adds the initial node, with members including all the nodes in the data_graph

        # self.update_clustering()  # initially everything is in the same cluster

    def add_edge(self, u, v, attr_dict=None, **attr):
        super(ClusterGraph, self).add_edge(u, v, attr_dict=None, **attr)

    def add_node(self, n, attr_dict=None, **attr):
        super(ClusterGraph, self).add_node(n, attr_dict=None, **attr)

        if 'members' in attr:
            self.node[n]['members'] = set(attr['members'])
            members = self.node[n]['members']
            for member in members:
                self.cluster_labels[member] = self.order()  # number of cluster nodes - keeps cluster labels consistent with the MATLAB code
            self.num_objects += len(members) # should it be len(attr['members']) instead?
        else:  # just one object
            self.num_objects += 1

    def remove_node(self, n):
        # assert self.has_node(n)
        self.num_objects -= 1
        for member in self.node[n]['members']:
            del self.cluster_labels[member]
            self.num_objects -= 1
        super(ClusterGraph, self).remove_node(n)


    def update_clustering(self, nodes=None) -> None:
        '''
        initializes the clustering
        :return:
        '''
        if nodes is None:
            nodes = self.nodes()
        for cnode in sorted(nodes):
            for member in self.node[cnode]['members']:
                self.cluster_labels[member] = cnode

    def add_cluster_member(self, cnode: int, member: int) -> None:
        assert self.has_node(cnode)
        self.node[cnode]['members'].add(member)
        self.cluster_labels[member] = self.cluster_labels[cnode]
        # self.update_clustering([cnode])


    def get_seed_pairs(self, cnode: int, params: Parameters) -> List[Tuple[Any, ...]]:
        assert self.has_node(cnode)
        members = self.node[cnode]['members']
        # if small enough, try all possiblec combinations
        if len(members) < 5:
            seed_pairs = list(combinations(members, 2))
        else:
            seed_pairs = []
            for member in members:
                pair = random.choice([m for m in members if m != member])
                seed_pairs.append((member, pair))

        if cnode < 0  or params.struct_name != 'partition' and self.num_objects > 1:
            pairs = deepcopy(seed_pairs)
            for a, b in pairs:
                seed_pairs.append((b, a))

        return seed_pairs


    def get_best_split(self, parent_node: int, params: Parameters) -> Tuple[Any, float, Set[int], Set[int]]:
        '''
        splits members of parent node into two sets
        :param members:
        :return: max_likelihood, set_1, set_2
        '''
        assert self.has_node(parent_node)

        members = self.node[parent_node]['members']
        print(f'Splitting cluster node {parent_node} ({members})')

        seed_pairs = self.get_seed_pairs(parent_node, params)
        # list of 3-tuples - (cluster_graph, likelihood, parts)
        cluster_graphs_likelihoods_and_partitions = []


        for seed_1, seed_2 in seed_pairs:  # try each seed pair
            members = self.node[parent_node]['members']
            assert seed_1 in members
            assert seed_2 in members

            # cluster_graph_copy = self.make_copy()
            cluster_graph_copy = deepcopy(self)
            data_graph = deepcopy(self.data_graph)
            # cluster_graph_copy.remove_node(parent_node)

            ## replace this with a call to node_split
            # cluster_graph_copy.add_node(seed_1, members={seed_1})
            # cluster_graph_copy.add_node(seed_2, members={seed_2})
            cluster_graph_copy.split_node(parent_node, seeds=(seed_1, seed_2))

            part_1 = {seed_1}
            part_2 = {seed_2}


            # check if the seeds have edges between them in the data graph - both directions
            # if data_graph.has_edge(seed_1, seed_2):
            #     cluster_graph_copy.add_edge(seed_1, seed_2)
            # if self.data_graph.has_edge(seed_2, seed_1):
            #     cluster_graph_copy.add_edge(seed_2, seed_1)

            # cluster_graph_copy.update_clustering()

            likelihood, cluster_graph_copy = graph_likelihood(data_graph=data_graph,
                                                              cluster_graph=cluster_graph_copy,
                                                              params=params)
            likelihood += graph_prior(cluster_graph_copy, params)  # this matches with the matlab scores
            # print('likelihood', likelihood)

            # set of member nodes
            members = members - {seed_1, seed_2}  # remove seeds from consideration
            max_likelihood = None
            for member in sorted(members):
                # try putting member into both clusters
                cluster_graph_1 = deepcopy(cluster_graph_copy)
                cluster_graph_2 = deepcopy(cluster_graph_copy)

                cluster_graph_1.add_cluster_member(seed_1, member)
                cluster_graph_2.add_cluster_member(seed_2, member)

                cluster_graph_1_likelihood, cluster_graph_1_new = graph_likelihood(data_graph=data_graph,
                                                                                   cluster_graph=cluster_graph_1, params=params)
                cluster_graph_1_likelihood += graph_prior(cluster_graph=cluster_graph_1, params=params)

                cluster_graph_2_likelihood, cluster_graph_2_new = graph_likelihood(data_graph=data_graph,
                                                                                   cluster_graph=cluster_graph_2,
                                                                                   params=params)
                cluster_graph_2_likelihood += graph_prior(cluster_graph=cluster_graph_2, params=params)

                print(f'{member}: graph1 like: {round(cluster_graph_1_likelihood, 4)}, graph2 like: {round(cluster_graph_2_likelihood, 4)}')

                if cluster_graph_1_likelihood >= cluster_graph_2_likelihood:
                    # print(f'{member} goes to cluster_1, members: {cluster_graph_1_new.cluster_labels.keys()}')
                    cluster_graph_copy = cluster_graph_1_new
                    part_1.add(member)
                    max_likelihood = cluster_graph_1_likelihood
                else:
                    # print(f'{member} goes to cluster_2, members: {cluster_graph_2_new.cluster_labels.keys()}')
                    cluster_graph_copy = cluster_graph_2_new
                    part_2.add(member)
                    max_likelihood = cluster_graph_2_likelihood

            # at this point, we have the best cluster_graph and the corresponding max log-likelihood for each seed-pair
            print(f'seeds: {seed_1, seed_2}, parts: {part_1}, {part_2}, likelihood: {round(max_likelihood, 3)}\n')
            cluster_graphs_likelihoods_and_partitions.append((cluster_graph_copy, max_likelihood, (part_1, part_2)))

        # we have multiple cluster_graphs and likelihoods

        # TODO: figure out simplify_graph in descending order of log_likelihood
        for cluster_graph, _, _ in sorted(cluster_graphs_likelihoods_and_partitions, key=lambda x: x[1], reverse=True):
            simplify_graph(cluster_graph)
            if cluster_graph.order() > 1:  # if after simplifying it has > 1 clusters, break
                break

        best_cluster_graph, best_likelihood, best_parts = max(filter(lambda x: x[0].order() > 1,  # only consider cluster_graphs with > 1 clusters
                                                                     cluster_graphs_likelihoods_and_partitions),
                                                              key=lambda x: x[1])  # pick the one with the best likelihood
        best_part_1, best_part_2 = best_parts
        print(f'best parts: {best_part_1, best_part_2}, likelihood: {best_likelihood}\n')

        assert len(best_part_1) > 0 and len(best_part_2) > 0, f'cannot split node {parent_node}'
        return best_cluster_graph, best_likelihood, best_part_1, best_part_2

    def split_node(self, parent_node: int, seeds: Tuple[int, int]) -> None:
        '''
        splits parent node into two children;
        :param parent_node:
        :param split_members:
        :return:
        '''
        assert self.has_node(parent_node), f'node {parent_node} not in cluster graph'
        incoming_nodes = set(self.predecessors(parent_node))  # end-points of the incoming red edges to the parent node
        outgoing_nodes = set(self.successors(parent_node))  # end-points of outgoing red edges to the parent node
        self.remove_node(parent_node)  # remove the parent node from the graph

        # adding the two new child nodes
        child_1, child_2 = seeds

        self.add_node(child_1, members={child_1})
        self.add_node(child_2, members={child_2})


        if self.struct_type == 'partition':
            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, child_1)
                self.add_edge(incoming_node, child_2)

            for outgoing_node in outgoing_nodes:
                self.add_edge(child_1, outgoing_node)
                self.add_edge(child_2, outgoing_node)

        elif self.struct_type == 'chain' or self.struct_type == 'ring':  # chains and rings have the same production rule
            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, child_1)

            self.add_edge(child_1, child_2)

            for outgoing_node in outgoing_nodes:
                self.add_edge(child_2, outgoing_node)

        elif self.struct_type == 'order':
            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, child_1)
                self.add_edge(incoming_node, child_2)

            self.add_edge(child_1, child_2)

            for outgoing_node in outgoing_nodes:
                self.add_edge(child_1, outgoing_node)
                self.add_edge(child_2, outgoing_node)

        elif self.struct_type == 'hierarchy':
            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, child_1)

            self.add_edge(child_1, child_2)

            for outgoing_node in outgoing_nodes:
                self.add_edge(child_1, outgoing_node)

        elif self.struct_type == 'tree':
            empty_node = f'empty_{parent_node}'
            self.add_node(empty_node, members={})
            self.number_empty_nodes += 1

            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, empty_node)

            self.add_edge(empty_node, child_1)
            self.add_edge(empty_node, child_2)

        else:
            raise NotImplementedError(f'Structure {self.struct_type} not yet implemented')


def simplify_graph(cluster_graph: ClusterGraph) -> None:
    ''' removes nodes if they meet the one of the following:
     remove:
     1. dangling cluster nodes: not an object node but has 0-1 cluster neighbors
     2. any cluster node with exactly two neighbors, one of which is a cluster node
    '''
    # notes: for trees only, there is a third check, not currently implemented
    # also under case 2, there's a tree subcase, not currently implemented
    check_dangling_and_empty_nodes = True
    # check_empty_nodes = True

    while check_dangling_and_empty_nodes:  # or check_empty_nodes:
        # dangling nodes: unoccupied node with 0 or 1
        remove_nodes = set()
        for node in cluster_graph.nodes():
            if len(cluster_graph.node[node]['members']) == 0 and len(cluster_graph.neighbors(node)) <= 2:  # TODO: this covers both cases
                remove_nodes.add(node)

        if len(remove_nodes) == 0:
            check_dangling_and_empty_nodes = False

        for node in remove_nodes:
            cluster_graph.remove_node(node)
    return
        # # empty nodes
        # remove_nodes = set()
        # for node in cluster_graph.nodes():
        #     if len(cluster_graph[node]['members']) == 0 and len(cluster_graph.adj[node]) == 2:
        #         remove_nodes.add(node)
        #
        # if len(remove_nodes) == 0:
        #     check_empty_nodes = False
        # cluster_graph.remove_nodes_from(remove_nodes)


def main():
    g = nx.Graph()
    # g.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (7, 8), (7, 9), (8, 9), (10, 11), (10, 12),
    #                   (11, 12), (3, 5), (6, 8), (9, 11)])
    g.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (3, 5)])

    g = nx.DiGraph(g)
    g.name = 'prisoners'

    params = Parameters()
    cluster_graph = ClusterGraph(struct_type='chain', dataset='test', data_graph=g)

    params.set_runtime_parameters(data_graph=g, struct_name=cluster_graph.struct_type)

    print(cluster_graph.get_best_split(0, params))
    # cluster_graph.add_edge('a', 'b')
    # print(cluster_graph.edges())


if __name__ == '__main__':
    main()
