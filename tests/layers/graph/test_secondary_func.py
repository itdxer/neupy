from collections import OrderedDict

from neupy.layers.base import (
    find_outputs_in_graph,
    topological_sort,
    filter_graph,
    is_cyclic,
    lazy_property,
)

from base import BaseTestCase


class ExtraFunctionsTestCase(BaseTestCase):
    def setUp(self):
        super(ExtraFunctionsTestCase, self).setUp()

        # 1 - 2 - 4 - 5
        # 3 _/     \_ 6 - 7
        self.graph = OrderedDict([
            (1, [2]),
            (3, [2]),
            (2, [4]),
            (4, [5, 6]),
            (6, [7]),
            (5, []),
            (7, []),
        ])

    def test_is_cyclic(self):
        self.assertTrue(is_cyclic({1: [2], 2: [3], 3: [1]}))
        self.assertTrue(is_cyclic({1: [2], 2: [3, 1], 3: []}))

        self.assertFalse(is_cyclic({1: [2], 2: [3, 4, 5], 3: [4, 5]}))
        self.assertFalse(is_cyclic({1: [2], 2: [3], 3: [4]}))
        self.assertFalse(is_cyclic(self.graph))

    def test_lazy_property(self):
        class SomeClass(object):
            @lazy_property
            def foo(self):
                return [1, 2, 3]

        a = SomeClass()
        self.assertIs(a.foo, a.foo)

    def test_filter_graph(self):
        filtered_graph = filter_graph({
            1: [2, 3],
            2: [3, 4],
            3: [5],
            4: [5],
            5: [],
        }, include_keys=[2, 3, 4])
        self.assertDictEqual(filtered_graph, {
            2: [3, 4],
            3: [],
            4: [],
        })

    def test_topological_sort(self):
        sorted_nodes = topological_sort(self.graph)
        self.assertSequenceEqual(sorted_nodes, [5, 7, 6, 4, 2, 1, 3])

    def test_find_outputs_in_graph(self):
        outputs = find_outputs_in_graph(self.graph)
        self.assertSequenceEqual(outputs, [5, 7])
