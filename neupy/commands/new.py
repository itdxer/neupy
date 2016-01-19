import os

from neupy.helpers.terminal import red


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_FOLDER_PATH = os.path.join(CURRENT_DIR, 'new_project_template')

MIN_COOKIECUTTER_VER = '0.8.0'


def run():
    try:
        from cookiecutter.main import cookiecutter
    except ImportError:
        print(red("Install `cookiecutter` >= {} before run "
                  "this command.\n".format(MIN_COOKIECUTTER_VER)))
        print("Copy/paste/run command below:")
        print("pip install cookiecutter>{}".format(MIN_COOKIECUTTER_VER))
        return

    cookiecutter(TEMPLATE_FOLDER_PATH)
