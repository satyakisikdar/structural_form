import networkx as nx

class ClusterGraph(nx.DiGraph):
    '''
    Structure for Cluster Graph where each node is a ClusterNode object
    '''
    def __init__(self, struct_type):
        nx.DiGraph.__init__(self)
        self.name = struct_type

    def split_score(self, child_1, child_2):
        '''
        calculates the log probability of the split of nodes child_1 and child_2
        :return: the log probability of the split
        '''
        return 0

    def get_split(self, parent_node):
        '''
        splits members into two sets
        :param members:
        :return: set_1, set_2
        '''
        members = self.node[parent_node]['members']
        part1 = members[: len(members)//2]
        part2 = members[len(members)//2: ]
        # disallow empty splits
        assert len(part1) > 0 and len(part2) > 0, f'cannot split node {parent_node}'
        return part1, part2

    def split_node(self, parent_node):
        '''
        splits parent node into two children;
        :param parent_node:
        :param split_members:
        :return:
        '''
        assert parent_node in self.nodes, f'node {parent_node} not in cluster graph'

        incoming_nodes = set(self.predecessors(parent_node))  # end-points of the incoming red edges to the parent node
        outgoing_nodes = set(self.successors(parent_node))  # end-points of outgoing red edges to the parent node

        members_1, members_2 = self.get_split(parent_node)

        # adding the two new child nodes
        child_1 = f'{parent_node}_1'
        child_2 = f'{parent_node}_2'
        self.add_node(child_1, members=members_1)
        self.add_node(child_2, members=members_2)

        if self.name == 'partition':
            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, child_1)
                self.add_edge(incoming_node, child_2)

            for outgoing_node in outgoing_nodes:
                self.add_edge(child_1, outgoing_node)
                self.add_edge(child_2, outgoing_node)

        elif self.name == 'chain' or self.name == 'ring':  # chains and rings have the same production rule
            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, child_1)

            self.add_edge(child_1, child_2)

            for outgoing_node in outgoing_nodes:
                self.add_edge(child_2, outgoing_node)

        elif self.name == 'order':
            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, child_1)
                self.add_edge(incoming_node, child_2)

            self.add_edge(child_1, child_2)

            for outgoing_node in outgoing_nodes:
                self.add_edge(child_1, outgoing_node)
                self.add_edge(child_2, outgoing_node)

        elif self.name == 'hierarchy':
            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, child_1)

            self.add_edge(child_1, child_2)

            for outgoing_node in outgoing_nodes:
                self.add_edge(child_1, outgoing_node)

        elif self.name == 'tree':
            dummy_node = ClusterNode(id=f'dummy_{parent_node.id}')
            self.add_node(dummy_node)

            for incoming_node in incoming_nodes:
                self.add_edge(incoming_node, dummy_node)

            self.add_edge(dummy_node, child_1)
            self.add_edge(dummy_node, child_2)

        else:
            raise NotImplementedError(f'Structure {self.name} not yet implemented')

        self.remove_node(parent_node)  # remove the parent node from the graph


def main():
    cluster_graph = ClusterGraph(struct_type='order')
    cluster_graph.add_node('0', members=list(range(10)))

    print(cluster_graph.nodes(data=True))
    cluster_graph.split_node('0')
    print()
    cluster_graph.split_node('0_2')
    print()
    cluster_graph.split_node('0_2_1')
    print()

if __name__ == '__main__':
    main()