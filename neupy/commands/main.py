import os
import importlib

import baker


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

manager = baker.Baker()


for filename in os.listdir(CURRENT_DIR):
    if filename not in ('__init__.py', 'main.py') and filename.endswith('py'):
        module_name = filename.rsplit('.')[0]
        module = importlib.import_module(
            'neupy.commands.{}'.format(module_name)
        )
        manager.command(name=module_name)(module.run)


def main():
    manager.run()
