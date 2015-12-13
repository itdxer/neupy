import re
from functools import reduce
from abc import ABCMeta

from six import with_metaclass

from neupy.utils import AttributeKeyDict


__all__ = ("SharedDocsMeta", "SharedDocs", "SharedDocsException",
           "SharedDocsABCMeta")


def merge_dicts(left_dict, right_dict):
    return dict(left_dict, **right_dict)


def find_numpy_doc_indent(docs):
    indent_detector = re.compile(r"(?P<indent>\ *)(?P<dashes>-{3,})")
    indent_info = indent_detector.findall(docs)

    if indent_info:
        indent, _ = indent_info[0]
        return indent


class SharedDocsException(Exception):
    """ Exception that help identify problems related to shared
    documentation.
    """


class SharedDocsMeta(type):
    """ Meta class for shared documentation. This class conatains main
    functionality that help inherit parameters and methods descriptions
    from parent classes. This class automaticaly format class documentation
    using basic python format syntax for objects.
    """

    def __new__(cls, clsname, bases, attrs):
        new_class = super(SharedDocsMeta, cls).__new__(cls, clsname, bases,
                                                       attrs)

        if new_class.__doc__ is None:
            return new_class

        doc_indent = find_numpy_doc_indent(new_class.__doc__)
        if doc_indent is None:
            return new_class

        parse_parameters = re.compile(
            r"(?P<name>\w+?)\s*\:\s*(?P<type>[^\n]+)"
            r"((?P<description>\n{indent}\ +[^\n]+)*)"
            "".format(indent=doc_indent)
        )
        parse_methods = re.compile(
            r"(?P<name>\w+?)(\((.+?)?\))"
            r"((?P<description>\n{indent}\ +[^\n]+)*)"
            "".format(indent=doc_indent)
        )

        parameters = {}
        parent_classes = new_class.__mro__

        for parent_class in parent_classes:
            parent_docs = parent_class.__doc__

            if parent_docs is None:
                continue

            parent_name = parent_class.__name__
            parent_params = parameters[parent_name] = AttributeKeyDict()

            doc_parameters = parse_parameters.findall(parent_docs)
            for name, type_, desc, _ in doc_parameters:
                parent_params[name] = "{} : {}{}".format(name, type_, desc)

            doc_methods = parse_methods.findall(parent_docs)
            for name, func_params, _, desc, _ in doc_methods:
                parent_params[name] = ''.join([name, func_params, desc])

        # TODO: after refatoring should remove old style shared documentaion

        # Collect parameter `shared_docs` for all MRO classes and
        # combine them in one big dictionary
        shared_docs = [getattr(b, 'shared_docs', {}) for b in parent_classes]
        all_params = reduce(merge_dicts, shared_docs, docs)
        parameters = merge_dicts(parameters, all_params)

        try:
            new_class.__doc__ = new_class.__doc__.format(**parameters)
        except ValueError as e:
            raise SharedDocsException("Can't format documentation for class "
                                      "`{}`. Catched exception: {}"
                                      "".format(new_class.__name__, e))

        return new_class


class SharedDocsABCMeta(SharedDocsMeta, ABCMeta):
    """ Meta class that combine ``SharedDocsMeta`` and ``ABCMeta``
    meta classes.
    """


class SharedDocs(with_metaclass(SharedDocsMeta)):
    """ Main class that provide with shared documentation functionality.
    """


docs = {
    # ------------------------------------ #
    #                Methods               #
    # ------------------------------------ #

    "last_error": """
    """,
    "plot_errors": """
    """,
    "predict": """
    """,
    "predict_raw": """predict_raw(input_data)
        Make a raw prediction. Ignore any post processing results related
        to the final output layer.
    """,
    "fit": """
    """,

    # ------------------------------------ #
    #             Train Methods            #
    # ------------------------------------ #

    "supervised_train": """
    """,
    "supervised_train_epochs": """train(input_data, target_data, epochs=100):
        Trains network with fixed number of epochs.
    """,
    "unsupervised_train_epochs": """train(input_train, epochs=100):
        Trains network with fixed number of epochs.
    """,
    "unsupervised_train_epsilon": """train(input_train, epsilon=1e-5, \
    epochs=100):
        Trains network until it converge. Parameter ``epochs`` control
        maximum number of iterations, just to make sure that network will
        stop training procedure if it can't converge.
    """,
    "supervised_train_lazy": """
    """,

    # ------------------------------------ #
    #              Parameters              #
    # ------------------------------------ #

    "verbose": """
    """,
    "step": """
    """,
    "show_epoch": """
    """,
    "shuffle_data": """
    """,
    "error": """
    """,
    "epoch_end_signal": """
    """,
    "train_end_signal": """
    """,

    # ------------------------------------ #
    #                 Steps                #
    # ------------------------------------ #
    "first_step": """first_step : float
        Contains initialized step value.
    """,
    "steps": """steps : list of float
        List of steps in the same order as the network layers.
        By default all values are equal to ``step`` parameter.
    """,

    # ------------------------------------ #
    #               Warnings               #
    # ------------------------------------ #
    "bp_depending": """It works with any algorithm based on backpropagation. \
    Class can't work without it.
    """
}


# ------------------------------------ #
#         Complex parameters           #
# ------------------------------------ #


def joindocs(docs, docskeys):
    return ''.join([docs[key] for key in docskeys])


full_params_params = ()
docs.update({
    'full_params': joindocs(
        docs,
        [
            'step', 'show_epoch', 'shuffle_data', 'epoch_end_signal',
            'train_end_signal', 'verbose'
        ]
    ),
    'full_signals': joindocs(
        docs, ['epoch_end_signal', 'train_end_signal']
    ),
    'full_methods': joindocs(
        docs, ['fit', 'predict', 'last_error', 'plot_errors']
    )
})
