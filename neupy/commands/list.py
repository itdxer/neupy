import os
import sys
import inspect
import importlib

from neupy.helpers.terminal import green


sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
)


FILES = (
    ('Algorithms', (
        ('algorithms',),
    )),
    ('Functions', (
        ('network', 'errors'),
    )),
    ('Layers', (
        ('layers', 'layers'),
        ('layers', 'output'),
    ))
)


def print_with_indent(text, indent=0):
    print(' ' * indent, '-', text)


def show_module_classes(module):
    for object_name in dir(module):
        object_value = getattr(module, object_name)

        is_not_valid_name = object_name.startswith('__')

        if not (is_not_valid_name or inspect.ismodule(object_value)):
            yield object_name


def get_section(section_number):
    n_sections = len(FILES)

    if section_number > n_sections:
        raise ValueError(
            "There is no section number {}. You can check sections "
            "from 1 to {}".format(section_number, n_sections)
        )

    section_name, filepatterns = FILES[section_number - 1]
    print(green('{}. {}'.format(section_number, section_name)))

    for filepattern in filepatterns:
        module_name = '.'.join(['neupy'] + list(filepattern))

        print_with_indent(module_name, indent=2)
        module = importlib.import_module(module_name)

        if hasattr(module, '__all__'):
            special_objects = module.__all__
        else:
            special_objects = show_module_classes(module)

        for special_object in sorted(special_objects):
            print_with_indent(special_object, indent=4)

        print('')


def run(section=None):
    if section is not None:
        get_section(int(section))
    else:
        for i in range(1, len(FILES) + 1):
            get_section(i)
