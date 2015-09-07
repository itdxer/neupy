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
            noize = np.random.binomial(1, 0.55, bin_password_len)
            data.append(noize)

        dhnet = algorithms.DiscreteHopfieldNetwork(mode='full')
        dhnet.train(np.array(data))

        return dhnet

If you read :ref:`Discrete Hopfield Network tutorial <discrete-hopfield-network>`, you must know that if we add only one vector into the network we will get it dublicated in whole matrix.
To make it little bit secure we can add the noize into the network.
For this reason we define one additional parameter ``noize_level`` into the function.
We encode our password into the binary vector and save it into the ``data`` variable.
Next we using Binomial distribution generate random binary vectors where probability to get 1 in vector is equal to 55%.
Parameter ``noize_level`` just control number of noize vectors.

But why do we get random binary vector instead of decoded random word?
The problem in the similarity between vectors.
Let's check two approaches with `Hamming distance <https://en.wikipedia.org/wiki/Hamming_distance>`_.

.. code-block:: python

    import string
    import random

    def hamming_distance(left, right):
        left, right = np.array(left), np.array(right)
        if left.shape != right.shape:
            raise ValueError("Shapes are different")
        return (left != right).sum()

    def generate_password(min_length=5, max_length=30):
        symbols = list(
            string.ascii_letters +
            string.digits +
            string.punctuation
        )
        password_len = random.randint(min_length, max_length + 1)
        password = [np.random.choice(symbols) for _ in range(password_len)]
        return ''.join(password)


In addition I add function ``generate_password`` that we use to test distance between randomly generated words.

.. code-block:: python

    >>> hamming_distance(str2bin(generate_password(20, 20)),
    ...                  str2bin(generate_password(20, 20))))
    71

As we can see two random generated passwords are very similar to each other (approximetly 70% of bits).
But If we compare randomly generated password and random binary vector we will see the difference.

.. code-block:: python

    >>> hamming_distance(str2bin(generate_password(20, 20)),
    ...                  np.random.binomial(1, 0.55, 238))
    123

Hamming distance is bigger than in previous example.
Almost 52% of the bits are different.
The bigger difference between random binary vector and word is improve possibility to recover valid passowrd from the network.


Recover password from the network
---------------------------------


Test it using Monte Carlo
-------------------------


Possible Problems
-----------------

* Shifted words

* Small percent recovery from empty string


.. author:: default
.. categories:: none
.. tags:: memory, unsupervised
.. comments::
