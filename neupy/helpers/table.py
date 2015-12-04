import time
from operator import attrgetter
from abc import ABCMeta, abstractmethod

from six import with_metaclass


__all__ = ("TableDrawer", "Column", "TimeColumn", "FloatColumn",
           "TableDrawingError")


class TableDrawingError(AttributeError):
    """ Exception specific for ``TableDrawer`` class functionality.
    """


class Column(object):
    """ Simple column class that helps discribe structure for
    ``TableDrawer`` class instance.

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


class TimeColumn(Column):
    """ Columns useful for time formating from seconds to more
    informative and readable format.
    """

    def format_value(self, value):
        """
        Parameters
        ----------
        value : float
            Time range in seconds.

        Returns
        -------
        str
            Seconds formated in readable string format.

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


# TODO: After a few modifications class got different behaviour and
# its name doesn't describe behaviour correctly. Should change it to
# some better name.
class FloatColumn(Column):
    """ Class describe float column type.

    Parameters
    ----------
    places : int
        Float number rounding precision. Defaults to ``6``.
    """

    def __init__(self, places=6, *args, **kwargs):
        super(FloatColumn, self).__init__(*args, **kwargs)
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
        if not isinstance(value, (int, float)):
            return value

        if value > 100:
            return "~{:.0f}".format(value)

        return round(value, self.places)


class BaseState(with_metaclass(ABCMeta)):
    """ Base abstract class that identify all important methods for
    ``TableDrawer`` class states.

    Parameters
    ----------
    table : TableDrawer instance
        Accept summary table instance. State is able to control
        properties from main ``TableDrawer`` class instantance
    """

    def __init__(self, table):
        self.table = table

    def line(self):
        """ Draw ASCII line. Line width depence on the table
        column sizes.
        """
        self.table.stdout('-' * self.table.total_width)

    def message(self, text):
        """ Write additional message in table. All seperators between
        columns will be ignored.
        """
        # Exclude from total width 2 separators and 2 spaces near them
        formated_text = text.ljust(self.table.total_width - 4)
        self.table.stdout("| " + formated_text + " |")

    def rewrite(self):
        # TODO: Add functionality
        raise NotImplementedError()

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
    """ Identify active state for ``TableDrawer`` class instance.
    In this state summary table instance is able to show information
    in terminal.
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
        self.table.stdout("| " + " | ".join(formated_data) + " |")


class IdleState(BaseState):
    """ Identify idle state for ``TableDrawer`` class instance.
    In this state summary table instance isn't able to show information
    in terminal.
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
        self.table.stdout("| " + " | ".join(headers) + " |")

        self.line()

    def row(self, data):
        raise TableDrawingError("Table drawing already finished or "
                                "didn't started")


class TableDrawer(object):
    """ Build ASCII tables using simple structure.

    Parameters
    ----------
    *columns
        Table structure. Accept ``Column`` instance classes.
    stdout : func
        Function through which the message will be transmitted.
    """

    def __init__(self, *columns, stdout=print):
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
        return super(TableDrawer, self).__getattr__(attr)
