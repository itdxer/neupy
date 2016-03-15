from collections import namedtuple

from neupy.helpers import table

from utils import catch_stdout
from base import BaseTestCase


Case = namedtuple("Case", "input_value expected_output")

table_drawing_result = """
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
""".strip()


class TableColumnsTestCase(BaseTestCase):
    def test_base_column(self):
        col1 = table.Column(name="Test1")
        self.assertIsInstance(col1.format_value(6), str)

        col2 = table.Column(name="Test2", dtype=int)
        self.assertIsInstance(col2.format_value(6.0), int)

    def test_float_column(self):
        col1 = table.NumberColumn(name="Test1", places=2)
        self.assertEqual(col1.format_value(1 / 3.), "0.33")

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


class TableDrawerTestCase(BaseTestCase):
    def test_idle_state_raises(self):
        table_drawing = table.TableDrawer(
            table.Column("Col 1"),
            table.Column("Col 2"),
        )

        with self.assertRaises(table.TableDrawingError):
            table_drawing.finish()

        with self.assertRaises(table.TableDrawingError):
            table_drawing.row([1, 2])

    def test_drawing_state_raises(self):
        table_drawing = table.TableDrawer(
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
        table_drawing = table.TableDrawer(
            table.Column("Col 1"),
            table.Column("Col 2", dtype=float),
            table.Column("Col 3", width=10),
        )

        with catch_stdout() as out:
            table_drawing.start()

            table_drawing.row(['test', 33, 'val'])
            table_drawing.row(['test2', -3, 'val 2'])
            table_drawing.line()
            table_drawing.message("Warning message")
            table_drawing.line()
            table_drawing.row(['test3', 0, 'val 3'])

            table_drawing.finish()
            terminal_output = out.getvalue().strip()

        self.assertEqual(table_drawing_result,
                         terminal_output.replace('\r', ''))
