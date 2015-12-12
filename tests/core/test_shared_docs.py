from neupy.core.docs import SharedDocs

from base import BaseTestCase


class SharedDocsTestCase(BaseTestCase):
    def test_global_docs(self):
        pass

    def test_simple_case(self):
        class A(SharedDocs):
            """ Class A documentation.

            Parameters
            ----------
            var1 : int
                Var1 description.
                Defaults to ``2``.
            var2 : str
                Var2 description.
            test : complex or float

            Examples
            --------
            one-two-three
            """

        class B(A):
            """ Class B documentation.

            Parameters
            ----------
            {A.var1}
            {A.var2}
            {A.test}
            """

        self.assertIn("var1 : int", B.__doc__)
        self.assertIn("var2 : str", B.__doc__)
        self.assertIn("test : complex or float", B.__doc__)
        self.assertIn("Class B documentation", B.__doc__)

    def test_complex_class_inheritance(self):
        pass
