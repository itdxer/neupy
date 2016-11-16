from __future__ import print_function

import time
import textwrap
from operator import attrgetter
from abc import abstractmethod

import numpy as np
from six import with_metaclass

from neupy.utils import number_type
from neupy.core.docs import SharedDocs, SharedDocsABCMeta


__all__ = ("TableBuilder", "Column", "TimeColumn", "NumberColumn",
           "TableDrawingError")


class TableDrawingError(AttributeError):
    """
    Exception specific for ``TableBuilder`` class functionality.
    """


class Column(SharedDocs):
    """
    Simple column class that helps discribe structure for
    ``TableBuilder`` class instance.

    Parameters
    ----------
    name : str
        Column name. Value would be displayed in header. In case when
        ``width`` parameter equal to ``None``, string width will identify
        column width.
    dtype : object
        Column data format. Defaults to ``str``.
    width : int or None
        Column width. Defaults to ``None``.
    """

    def __init__(self, name, dtype=str, width=None):
        if width is None:
            width = len(name)

        self.name = name
        self.dtype = dtype
        self.width = width

    def format_value(self, value):
        """
        Convert input value to specified type

        Parameters
        ----------
        value : object

        Returns
        -------
        object
            Function return converted input value to specified
            data type.
        """
        return self.dtype(value)


def format_time(value):
    """
    Convert seconds to the value format that easy
    to understand.

    Parameters
    ----------
    value : float
        Time interval in seconds.

    Returns
    -------
    str

    Examples
    --------
    >>> col = TimeColumn("Time")
    >>> col.format_value(0.001)
    '1 ms'
    >>> col.format_value(0.5)
    '0.5 sec'
    >>> col.format_value(1.5)
    '1.5 sec'
    >>> col.format_value(15)
    '00:00:15'
    >>> col.format_value(15045)
    '04:10:45'
    """
    if value < 0.05:
        return "{} ms".format(round(value * 10 ** 3))

    elif value < 10:
        return "{} sec".format(round(value, 1))

    return time.strftime("%H:%M:%S", time.gmtime(value))


class TimeColumn(Column):
    """
    Columns useful for time formating from seconds to more
    informative and readable format.

    Parameters
    ----------
    {Column.name}
    {Column.dtype}
    {Column.width}
    """

    def format_value(self, value):
        return format_time(value)


class NumberColumn(Column):
    """
    Class describe float column type.

    Parameters
    ----------
    places : int
        Float number rounding precision. Defaults to ``6``.
    {Column.name}
    {Column.dtype}
    {Column.width}
    """

    def __init__(self, places=6, *args, **kwargs):
        super(NumberColumn, self).__init__(*args, **kwargs)
        self.places = places

    def format_value(self, value):
        """
        Round a number to a given precision in decimal digits

        Parameters
        ----------
        value : float

        Returns
        -------
        float
            Rounded input value.
        """
        if not isinstance(value, number_type):
            return value
        return "{value:.{places}g}".format(value=value, places=self.places)


class BaseState(with_metaclass(SharedDocsABCMeta)):
    """
    Base abstract class that identify all important methods for
    ``TableBuilder`` class states.

    Parameters
    ----------
    table : TableBuilder instance
        Accept summary table instance. State is able to control
        properties from main ``TableBuilder`` class instantance
    """

    def __init__(self, table):
        self.table = table

    def line(self):
        """
        Draw ASCII line. Line width depends on the table
        column sizes.
        """
        self.table.stdout('\r' + '-' * self.table.total_width)

    def message(self, text):
        """
        Write additional message in table. All seperators
        between columns will be ignored.
        """
        self.line()
        # Excluding from the total width 2 symbols related to
        # the separators near the table edges and 2 symbols
        # related to the spaces near these edges
        max_line_width = self.table.total_width - 4

        for text_row in textwrap.wrap(text, max_line_width):
            formated_text = text_row.ljust(max_line_width)
            self.table.stdout("\r| " + formated_text + " |")

        self.line()

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        raise NotImplementedError

    @abstractmethod
    def header(self):
        raise NotImplementedError

    @abstractmethod
    def row(self, data):
        raise NotImplementedError


class DrawingState(BaseState):
    """
    Identify active state for ``TableBuilder`` class instance.
    In this state summary table instance is able to show information
    in terminal.

    Parameters
    ----------
    {BaseState.table}
    """

    def start(self):
        raise TableDrawingError("Table drawing already started")

    def finish(self):
        self.line()
        self.table.state = IdleState(self.table)

    def header(self):
        raise TableDrawingError("Header already drawn")

    def row(self, data):
        formated_data = []
        for val, column in zip(data, self.table.columns):
            val = column.format_value(val)
            cell_value = str(val).ljust(column.width)
            formated_data.append(cell_value)
        self.table.stdout("\r| " + " | ".join(formated_data) + " |")


class IdleState(BaseState):
    """
    Identify idle state for ``TableBuilder`` class instance.
    In this state summary table instance isn't able to show information
    in terminal.

    Parameters
    ----------
    {BaseState.table}
    """

    def start(self):
        self.header()
        self.table.state = DrawingState(self.table)

    def finish(self):
        raise TableDrawingError("Table drawing already finished or "
                                "didn't started")

    def header(self):
        self.line()

        headers = []
        for column in self.table.columns:
            header_name = str(column.name).ljust(column.width)
            headers.append(header_name)
        self.table.stdout("\r| " + " | ".join(headers) + " |")

        self.line()

    def row(self, data):
        raise TableDrawingError("Table drawing already finished or "
                                "didn't started")


class TableBuilder(SharedDocs):
    """
    Build ASCII tables using simple structure.

    Parameters
    ----------
    *columns
        Table structure. Accept ``Column`` instance classes.
    stdout : func
        Function through which the message will be transmitted.
    """
    def __init__(self, *columns, **kwargs):
        # In Python 2 doesn't work syntax like
        # def __init__(self, *columns, stdout=print):
        # Code below implements the same.
        valid_kwargs = ['stdout']
        stdout = kwargs.get('stdout', print)

        if any(kwarg not in valid_kwargs for kwarg in kwargs):
            raise ValueError("Invalid keyword arguments. Available "
                             "only: {}".format(valid_kwargs))

        for column in columns:
            if not isinstance(column, Column):
                raise TypeError("Column should be an instance of "
                                "the `Column` class")

        self.columns = columns
        self.stdout = stdout
        self.state = IdleState(self)

        text_width = sum(map(attrgetter('width'), columns))
        n_columns = len(columns)
        n_separators = n_columns + 1
        n_margins = 2 * n_columns

        self.total_width = text_width + n_separators + n_margins

    @classmethod
    def show_full_table(cls, columns, values, **kwargs):
        """
        Shows full table. This method is useful in case if all
        table values are available and we can just show them all
        in one table without interations.

        Parameters
        ----------
        columns : list
            List of columns.
        values : list of list or tuple
            List of values. Each element should be a list or
            tuple that contains column values in the same order
            as defined in the ``columns`` parameter.
        **kwargs
            Arguments for the ``TableBuilder`` class
            initialization.
        """
        values_length = []
        for row_values in values:
            row_values_length = [len(str(value)) for value in row_values]
            values_length.append(row_values_length)

        columns_width = np.array(values_length).max(axis=0)
        for column, proposed_column_width in zip(columns, columns_width):
            column.width = max(column.width, proposed_column_width)

        table_builder = cls(*columns, **kwargs)
        table_builder.start()

        for row_values in values:
            table_builder.row(row_values)

        table_builder.finish()

    def __getattr__(self, attr):
        if attr not in self.__dict__ and hasattr(self.state, attr):
            return getattr(self.state, attr)
        return super(TableBuilder, self).__getattribute__(attr)
