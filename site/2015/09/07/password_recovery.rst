.. _password-recovery:

Password recovery
=================

.. contents::

At this tutorial we are going to build a simple network that will recover password from the broken one.
If you don't familiar with :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` algorithm, you can read :ref:`this tutorial <discrete-hopfield-network>`, but it is not realy nesessary.

Data transformation
-------------------

Before build network that will save and recover password we must make transformations for the input and output data.
Lets assume that we are use only ascii symbols.
We must make binary vector with a fixed length from a word.
Let's define a functio that transform string to the binary list.

.. code-block:: python

    import binascii

    def str2bin(text, max_length=30, encoding='ascii'):
        if len(text) > max_length:
            raise ValueError("Text can't contains more "
                             "than {} symbols".format(max_length))

        text = text.rjust(max_length)
        byte_text = bytearray(text, encoding)
        binary_string = bin(int(binascii.hexlify(byte_text), 16))

        # Remove 0b symbols from string
        binary_array = map(int, binary_string[2:])

        # Python cut leading zeros from the beggining,
        # so we need put them back
        letter_bin_len = math.ceil((len(binary_string) - 2) / max_length)
        valid_length = letter_bin_len * max_length
        bin_vector = list(binary_array)

        n_leading_zeros = valid_length - len(bin_vector)
        leading_zeros = list(repeat(0, times=n_leading_zeros))

        return leading_zeros + bin_vector

Let's test it with a word ``test``.

.. code-block:: python

    >>> str2bin("test", max_length=5)
    [0, 0, 1, 0, 0, 0, 0, 0, 0 ...
    >>> len(str2bin("test", max_length=5))
    40


Ascii use 8 bit per symbol and we set up 5 symbols per word, so our vector length must always equal be to 40.
If word contains less symbols than you set up in ``max_length`` parameter, function will fill spaces at the beggining of the word.
As you can see first 8 symbols from output have form ``00100000`` which is space value from the ascii table.

Now we must add another function that transform binary vector into string.

.. code-block:: python

    def bin2str(array, encoding='ascii'):
        string_array = map(str, array)
        binary_string = ''.join(string_array)
        binary_string = '0b' + binary_string
        hex_string = bytearray(hex(int(binary_string, 2)), encoding)

        raw_text = binascii.unhexlify(hex_string[2:])
        return raw_text.lstrip().decode(encoding)

And if we test it we get word ``test`` back.

.. code-block:: python

    >>> bin2str(str2bin("test", max_length=5))
    'test'

Save password into the network
------------------------------

Now we are ready save password into the network.
For this task we are going to define another function.
Let's define it and later we will check it step by step.

.. code-block:: python

    import numpy as np
    from neupy import algorithms

    def save_password(real_password, noize_level=5):
        if noize_level < 1:
            raise ValueError("`noize_level` must be equal or greater than 1.")

        binary_password = str2bin(real_password)
        bin_password_len = len(binary_password)

        data = [binary_password]

        for _ in range(noize_level):
            # The farther from the 0.5 value the less likely
            # password recovery
            noize = np.random.binomial(1, 0.55, len(str2bin(real_password)))
            data.append(noize)

        dhnet = algorithms.DiscreteHopfieldNetwork(mode='full')
        dhnet.train(np.array(data))

        return dhnet


Recover password from the network
----------------------------------


Problems
--------

* Shifted words

* Small percent recovery from empty string


.. author:: default
.. categories:: none
.. tags:: memory, unsupervised
.. comments::
