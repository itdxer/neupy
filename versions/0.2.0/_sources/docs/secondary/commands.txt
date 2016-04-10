Commands
========

NeuPy library provides some simple terminal commands which will help you with development process.

help
----

You can always check the list of available commands from terminal.

.. code-block:: bash

    $ neupy -h

Also you can check help description for specific command.

.. code-block:: bash

    $ neupy list -h

list
----

Display all available algorithms, layers and error function in the library.
More information you can also find on `this page <../algorithms.html>`_

.. code-block:: bash

    $ neupy list

You can check just one section from this list.

.. code-block:: bash

    $ neupy list --section 1

The same command but shorter version.

.. code-block:: bash

    $ neupy list 1

new
---

This command will create new project folder.

.. code-block:: bash

    $ neupy new
    project_name (default is "application")? testproject
    $ ls
    testproject
