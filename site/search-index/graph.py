from collections import OrderedDict


class DirectedGraph(object):
    def __init__(self):
        self.graph = OrderedDict()

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, node_1, node_2):
        self.add_node(node_1)
        self.add_node(node_2)

        self.graph[node_1].append(node_2)

    def __iter__(self):
        for from_node, to_nodes in self.graph.items():
            yield from_node, to_nodes

    def __len__(self):
        return len(self.graph)

    @property
    def edges(self):
        return list(self.graph.keys())
