from __future__ import print_function

import time
import textwrap
from operator import attrgetter
from abc import abstractmethod

import numpy as np
from six import with_metaclass

from neupy.core.docs import SharedDocs, SharedDocsABCMeta


__all__ = ("TableBuilder", "Column", "TimeColumn", "NumberColumn",
           "TableDrawingError")


class TableDrawingError(AttributeError):
    """ Exception specific for ``TableBuilder`` class functionality.
    """


class Column(SharedDocs):
    """ Simple column class that helps discribe structure for
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
        """ Convert input value to specified type

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
    """ Convert seconds to the value format that easy
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
    """ Columns useful for time formating from seconds to more
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
    """ Class describe float column type.

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
        """ Round a number to a given precision in decimal digits

        Parameters
        ----------
        value : float

        Returns
        -------
        float
            Rounded input value.
        """
        if not isinstance(value, (int, float, np.floating, np.integer)):
            return value

        if value > 100:
            return "~{:.0f}".format(value)

        return "{value:.{places}f}".format(value=value,
                                           places=self.places)


class BaseState(with_metaclass(SharedDocsABCMeta)):
    """ Base abstract class that identify all important methods for
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
        """ Draw ASCII line. Line width depence on the table
        column sizes.
        """
        self.table.stdout('\r' + '-' * self.table.total_width)

    def message(self, text):
        """ Write additional message in table. All seperators
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
        pass

    @abstractmethod
    def finish(self):
        pass

    @abstractmethod
    def header(self):
        pass

    @abstractmethod
    def row(self, data):
        pass


class DrawingState(BaseState):
    """ Identify active state for ``TableBuilder`` class instance.
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
    """ Identify idle state for ``TableBuilder`` class instance.
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
    """ Build ASCII tables using simple structure.

    Parameters
    ----------
    *columns
        Table structure. Accept ``Column`` instance classes.
    stdout : func
        Function through which the message will be transmitted.
    """
    def __init__(self, *columns, **kwargs):
        valid_kwargs = ['stdout']
        # In Python 2 doesn't work syntax like
        # def __init__(self, *columns, stdout=print):
        # Code below implements the same.
        stdout = kwargs.get('stdout', print)

        if any(kwarg not in valid_kwargs for kwarg in kwargs):
            raise ValueError("Invalid keyword arguments. Available "
                             "only: {}".format(valid_kwargs))

        for column in columns:
            if not isinstance(column, Column):
                raise TypeError("Column should be ``Column`` class "
                                "instance.")

        self.columns = columns
        self.stdout = stdout
        self.state = IdleState(self)

        text_width = sum(map(attrgetter('width'), columns))
        n_columns = len(columns)
        n_separators = n_columns + 1
        n_margins = 2 * n_columns

        self.total_width = text_width + n_separators + n_margins

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.state, attr)
        return super(TableBuilder, self).__getattr__(attr)
