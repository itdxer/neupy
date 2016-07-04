from neupy.core.docs import SharedDocs, shared_docs

from base import BaseTestCase


class SharedDocsTestCase(BaseTestCase):
    def test_simple_case(self):
        class A(SharedDocs):
            """
            Class A documentation.

            Parameters
            ----------
            var1 : int
                Var1 description.
                Defaults to ``2``.
            var2 : str
                Var2 description.
            test : complex or float
            """

        class B(A):
            """
            Class B documentation.

            Parameters
            ----------
            {A.var1}
            {A.var2}
            {A.test}
            """

        self.assertIn("Class B documentation", B.__doc__)

        self.assertIn("var1 : int", B.__doc__)
        self.assertIn("var2 : str", B.__doc__)
        self.assertIn("test : complex or float", B.__doc__)
        self.assertIn("Defaults to ``2``.", B.__doc__)

    def test_shared_methods(self):
        class A(SharedDocs):
            """
            Class A documentation.

            Methods
            -------
            foo()
                Foo description.
                Even more
            bar(params=True)
            double_row(param1=True, param2=True,\
            param3=True)
                Additional description for ``double_row``.

            Examples
            --------
            one-two-three
            """

        class B(A):
            """
            Class B documentation.

            Methods
            -------
            {A.foo}
            {A.bar}
            {A.double_row}
            """

        self.assertIn("Class B documentation", B.__doc__)

        self.assertIn("foo()", B.__doc__)
        self.assertIn("Foo description.", B.__doc__)
        self.assertIn("Even more", B.__doc__)
        self.assertIn("bar(params=True)", B.__doc__)

        # Check multi-row method
        self.assertIn("double_row(param1=True, param2=True,", B.__doc__)
        self.assertIn("param3=True)", B.__doc__)
        self.assertIn("Additional description for ``double_row``.", B.__doc__)

    def test_shared_warns(self):
        class A(SharedDocs):
            """
            Class A documentation.

            Warns
            -----
            Important warning.
            Additional information related to warning message.

            Examples
            --------
            Just to put sth before `Warns`
            """

        class B(A):
            """
            Class B documentation.

            Warns
            -----
            {A.Warns}
            """

        self.assertIn("Class B documentation", B.__doc__)

        self.assertIn("Important warning", B.__doc__)
        self.assertIn("related to warning message", B.__doc__)

        self.assertNotIn("Just to", B.__doc__)
        self.assertNotIn("Examples", B.__doc__)

    def test_complex_class_inheritance(self):
        class A(SharedDocs):
            """
            Class A documentation.

            Parameters
            ----------
            var_a : int
            var_x : str
            """

        class B(SharedDocs):
            """
            Class B documentation

            Parameters
            ----------
            var_b : int
            var_x : float
            """

        class C(A, B):
            """
            Class C documentation.

            Parameters
            ----------
            {A.var_a}
            {B.var_b}
            {A.var_x}
            """

        self.assertIn("Class C documentation", C.__doc__)

        self.assertIn("var_a : int", C.__doc__)
        self.assertIn("var_b : int", C.__doc__)
        self.assertIn("var_x : str", C.__doc__)

    # def test_share_all_parameters(self):
    #     class A(SharedDocs):
    #         """ Class A documentation.
    #
    #         Parameters
    #         ----------
    #         var_a : int
    #         var_x : str
    #         """
    #
    #     class B(A):
    #         """ Class B documentation
    #
    #         Parameters
    #         ----------
    #         {A.var_a}
    #         var_b : int
    #         var_x : float
    #         """
    #
    #     class C(B):
    #         """ Class C documentation.
    #
    #         Parameters
    #         ----------
    #         {B.Parameters}
    #         """
    #
    #     self.assertIn("Class C documentation", C.__doc__)
    #
    #     self.assertIn("var_a : int", C.__doc__)
    #     self.assertIn("var_b : int", C.__doc__)
    #     self.assertIn("var_x : float", C.__doc__)

    def test_shared_docs_between_functions(self):
        def function_a(x, y):
            """
            Function A documentation.

            Parameters
            ----------
            x : int
                First input varaible x.
            y : int
                Second input variable y.

            Returns
            -------
            int
                Output is equal to x + y.
            """

        @shared_docs(function_a)
        def function_b(x, y):
            """
            Function B documentation.

            Parameters
            ----------
            {function_a.x}
            {function_a.y}

            Returns
            -------
            int
                Output is equal to x * y.
            """

        docs = function_b.__doc__
        self.assertIn("Function B documentation", docs)

        self.assertIn("x : int", docs)
        self.assertIn("First input varaible x.", docs)
        self.assertIn("y : int", docs)
        self.assertIn("Second input variable y.", docs)
