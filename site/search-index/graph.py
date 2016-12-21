from collections import OrderedDict


class DirectedGraph(object):
    def __init__(self):
        self.graph = OrderedDict()

    def add_node(self, node):
        self.graph[node] = []

    def add_edge(self, node_1, node_2):
        if node_1 not in self.graph:
            self.add_node(node_1)

        if node_2 not in self.graph:
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
