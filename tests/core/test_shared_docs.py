from neupy.core.docs import SharedDocs, shared_docs, parse_variables_from_docs

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

    def test_inherit_method_docs(self):
        class A(SharedDocs):
            def method(self, x, y):
                """
                Class A method.

                Parameters
                ----------
                x : int
                    X variable.
                y : int
                    Y varaible.

                Returns
                -------
                int
                    x + y
                """
                return x + y

            def method2(self):
                pass

        class B(A):
            def method(self, x, y):
                # It's suppose to inherit documentation
                # from the A class
                return x + y

            def method2(self):
                pass

        self.assertEqual(B.method.__doc__, A.method.__doc__)
        self.assertIsNone(B.method2.__doc__)

        class C(A):
            def method(self, x, y):
                """
                Updated documentation for the method.
                """
                return x + y

        self.assertNotEqual(C.method.__doc__, A.method.__doc__)
        self.assertIn("Updated documentation for the method.",
                      C.method.__doc__)

    def test_disabled_inherit_method_docs(self):
        class A(SharedDocs):
            def method(self, x, y):
                """
                Method 1.
                """

        class B(A):
            inherit_method_docs = False

            def method(self, x, y):
                pass

        self.assertIsNone(B.method.__doc__)

        class C(A):
            inherit_method_docs = True

            def method(self, x, y):
                pass

        self.assertIsNotNone(C.method.__doc__)

    def test_args_and_kwargs_parameters(self):
        class A(SharedDocs):
            """
            Class A

            Parameters
            ----------
            *args
                Arguments.
            **kwargs
                Keyword Arguments.
            """

        class B(A):
            """
            Class B

            Parameters
            ----------
            {A.args}
            {A.kwargs}
            """

        docs = B.__doc__
        self.assertIn("Class B", docs)

        self.assertIn("*args", docs)
        self.assertNotIn("*args :", docs)
        self.assertIn("Arguments.", docs)

        self.assertIn("**kwargs", docs)
        self.assertNotIn("**kwargs :", docs)
        self.assertIn("Keyword Arguments.", docs)


class SharedDocsParseSectionsTestCase(BaseTestCase):
    def test_shared_warns_section(self):
        class A(SharedDocs):
            """
            Class A documentation.

            Warns
            -----
            Important warning.
            Additional information related to warning message.

            Examples
            --------
            Some section before `Warns`
            """

        class B(A):
            """
            Class B documentation.

            Warns
            -----
            {A.Warns}
            """

        docs = B.__doc__
        self.assertIn("Class B documentation", docs)

        self.assertIn("Important warning", docs)
        self.assertIn("related to warning message", docs)

        self.assertNotIn("Some section before", docs)
        self.assertNotIn("Examples", docs)

    def test_shared_returns_section(self):
        class A(SharedDocs):
            """
            A class

            Returns
            -------
            int
                Add two number together
            """

        class B(A):
            """
            B class

            Returns
            -------
            {A.Returns}
            """

            expected_doc = """
            B class

            Returns
            -------
            int
                Add two number together
            """

        self.assertEqual(B.__doc__, B.expected_doc)

    def test_shared_yields_section(self):
        def foo(x, y):
            """
            foo function

            Yields
            ------
            int
                Integer values.
            """

        @shared_docs(foo)
        def bar(x, y):
            """
            bar function

            Yields
            ------
            {foo.Yields}
            """

        def expected(x, y):
            """
            bar function

            Yields
            ------
            int
                Integer values.
            """

        self.assertEqual(bar.__doc__, expected.__doc__)

    def test_shared_raises_section(self):
        def foo(x, y):
            """
            foo function

            Raises
            ------
            ValueError
                Just raise it all the time
            """

        @shared_docs(foo)
        def bar(x, y):
            """
            bar function

            Raises
            ------
            {foo.Raises}
            """

        def expected(x, y):
            """
            bar function

            Raises
            ------
            ValueError
                Just raise it all the time
            """

        self.assertEqual(bar.__doc__, expected.__doc__)

    def test_shared_see_also(self):
        def foo(x, y):
            """
            foo function

            See Also
            --------
            foo1, foo2, foo3
            foo4
            """

        @shared_docs(foo)
        def bar(x, y):
            """
            bar function

            See Also
            --------
            {foo.See Also}
            """

        def expected(x, y):
            """
            bar function

            See Also
            --------
            foo1, foo2, foo3
            foo4
            """

        self.assertEqual(bar.__doc__, expected.__doc__)

    def test_share_all_parameters_section(self):
        class A(SharedDocs):
            """
            Class A documentation.

            Parameters
            ----------
            var_a : int
                Variable a

                * List element 1.

                * List element 2.
            var_x : str
                Variable x string
            """

        class B(A):
            """
            Class B documentation

            Parameters
            ----------
            {A.var_a}
            var_b : int
                Variable b
            var_x : float
                Variable x float
            """

        class C(B):
            """
            Class C documentation.

            Parameters
            ----------
            {B.Parameters}
            """

        class ExpectedDoc(object):
            """
            Class C documentation.

            Parameters
            ----------
            var_a : int
                Variable a

                * List element 1.

                * List element 2.
            var_b : int
                Variable b
            var_x : float
                Variable x float
            """

        self.assertEqual(C.__doc__, ExpectedDoc.__doc__)

    def test_shared_attributes_section(self):
        class A(SharedDocs):
            """
            Class A.

            Attributes
            ----------
            x : int
            y : str
                Short description.

            Examples
            --------
            >>> a = A()
            >>> a.x
            0
            """

        class B(A):
            """
            Class B.

            Attributes
            ----------
            {A.Attributes}

            Examples
            --------
            >>> b = B()
            >>> b.x
            1
            """

        class ExpectedDoc(object):
            """
            Class B.

            Attributes
            ----------
            x : int
            y : str
                Short description.

            Examples
            --------
            >>> b = B()
            >>> b.x
            1
            """

        self.assertEqual(B.__doc__, ExpectedDoc.__doc__)

    def test_shared_methods_section(self):
        class A(SharedDocs):
            """
            Class A.

            Methods
            -------
            add(x, y)
                Returns x + y
            mul(x, y)
                Returns x * y

            Warns
            -----
            Some warnings
            """

        class B(A):
            """
            Class B.

            Methods
            -------
            {A.Methods}
            """

        class ExpectedDoc(object):
            """
            Class B.

            Methods
            -------
            add(x, y)
                Returns x + y
            mul(x, y)
                Returns x * y
            """

        self.assertEqual(B.__doc__, ExpectedDoc.__doc__)

    def test_parse_variables_from_docs(self):
        self.assertEqual({}, parse_variables_from_docs([]))

        class A(object):
            pass

        self.assertEqual({}, parse_variables_from_docs([A]))
