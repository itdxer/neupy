import textwrap
from collections import namedtuple

from neupy.helpers import table

from utils import catch_stdout
from base import BaseTestCase


Case = namedtuple("Case", "input_value expected_output")


class TableColumnsTestCase(BaseTestCase):
    def test_base_column(self):
        col1 = table.Column(name="Test1")
        self.assertIsInstance(col1.format_value(6), str)

        col2 = table.Column(name="Test2", dtype=int)
        self.assertIsInstance(col2.format_value(6.0), int)

    def test_float_column(self):
        col1 = table.NumberColumn(name="Test1", places=2)

        test_cases = [
            Case(input_value=1 / 3., expected_output='0.33'),
            Case(input_value=30, expected_output="30"),
            Case(input_value=3000, expected_output="3e+03"),
            Case(input_value=3141592, expected_output="3.1e+06"),
            Case(input_value=0.000005123, expected_output="5.1e-06"),
        ]

        for test_case in test_cases:
            actual_output = col1.format_value(test_case.input_value)
            self.assertEqual(test_case.expected_output, actual_output)

    def test_time_column(self):
        test_cases = [
            Case(input_value=0.1, expected_output='0.1 sec'),
            Case(input_value=1.0, expected_output='1.0 sec'),
            Case(input_value=1.1234, expected_output='1.1 sec'),
            Case(input_value=9.99, expected_output='10.0 sec'),

            Case(input_value=10, expected_output='00:00:10'),
            Case(input_value=70, expected_output='00:01:10'),
            Case(input_value=3680, expected_output='01:01:20'),
        ]
        col1 = table.TimeColumn(name="Test1")

        for case in test_cases:
            self.assertEqual(col1.format_value(case.input_value),
                             case.expected_output)


class TableBuilderTestCase(BaseTestCase):
    def test_idle_state_raises(self):
        table_drawing = table.TableBuilder(
            table.Column("Col 1"),
            table.Column("Col 2"),
        )

        with self.assertRaises(table.TableDrawingError):
            table_drawing.finish()

        with self.assertRaises(table.TableDrawingError):
            table_drawing.row([1, 2])

    def test_drawing_state_raises(self):
        table_drawing = table.TableBuilder(
            table.Column("Col 1"),
            table.Column("Col 2"),
        )

        with catch_stdout():
            table_drawing.start()

            with self.assertRaises(table.TableDrawingError):
                table_drawing.start()

            with self.assertRaises(table.TableDrawingError):
                table_drawing.header()

    def test_table_drawing(self):
        table_drawing_result = textwrap.dedent("""
        ------------------------------
        | Col 1 | Col 2 | Col 3      |
        ------------------------------
        | test  | 33.0  | val        |
        | test2 | -3.0  | val 2      |
        ------------------------------
        | Warning message            |
        ------------------------------
        | test3 | 0.0   | val 3      |
        ------------------------------
        """).strip()

        table_drawing = table.TableBuilder(
            table.Column("Col 1"),
            table.Column("Col 2", dtype=float),
            table.Column("Col 3", width=10),
        )

        with catch_stdout() as out:
            table_drawing.start()

            table_drawing.row(['test', 33, 'val'])
            table_drawing.row(['test2', -3, 'val 2'])
            table_drawing.message("Warning message")
            table_drawing.row(['test3', 0, 'val 3'])

            table_drawing.finish()
            terminal_output = out.getvalue().strip()
            terminal_output = terminal_output.replace('\r', '')

        # Use assertTrue to make sure that it won't through
        # all variables in terminal in case of error
        self.assertTrue(table_drawing_result == terminal_output)

    def test_table_builder_exception(self):
        with self.assertRaises(ValueError):
            table.TableBuilder(invalid=True)

        with self.assertRaises(TypeError):
            table.TableBuilder(
                table.Column("Col 1"),
                'not a column',
            )

    def test_table_state_variables(self):
        table_builder = table.TableBuilder(table.Column("Col 1"))

        # use direct access to __getattr__ for code coverage
        self.assertEqual(len(table_builder.__getattr__('columns')), 1)
        # self.assertIn is not working here
        self.assertEqual(id(table_builder.__getattr__('row')),
                         id(table_builder.state.row))
