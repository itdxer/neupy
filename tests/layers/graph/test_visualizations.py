import mock
import tempfile
import graphviz

from neupy import layers

from base import BaseTestCase


class MockedDigraph(graphviz.Digraph):
    events = []

    def node(self, node_id, name):
        self.events.append('add_node')
        return super(MockedDigraph, self).node(node_id, name)

    def edge(self, node1_id, node2_id, *args, **kwargs):
        self.events.append('add_edge')
        return super(MockedDigraph, self).edge(
            node1_id, node2_id, *args, **kwargs)

    def render(self, filepath, *args, **kwargs):
        self.events.append(filepath)
        self.events.append('render')


@mock.patch('graphviz.Digraph', side_effect=MockedDigraph)
class GraphVisualizationTestCase(BaseTestCase):
    def setUp(self):
        super(GraphVisualizationTestCase, self).setUp()
        MockedDigraph.events = self.events = []

    def test_basic_visualization(self, _):
        network = layers.join(
            layers.Input(10),
            layers.Relu(5),
            layers.parallel(
                layers.Relu(2),
                layers.Relu(3),
            )
        )
        network.show()

        # 4 layers + 2 outputs
        self.assertEqual(6, self.events.count('add_node'))

        # 3 between layers + 2 between layers and extra output nodes
        self.assertEqual(5, self.events.count('add_edge'))

        # Always has to be called at the end
        self.assertEqual(1, self.events.count('render'))
        self.assertEqual(self.events[-1], 'render')

        filepath = tempfile.mktemp()
        network.show(filepath)
        self.assertEqual(self.events[-2], filepath)
