.. _password-recovery:

Password recovery
=================

.. contents::

At this tutorial we are going to build a simple network that will recover password from a broken one.
If you don't familiar with :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` algorithm, you can read :ref:`this tutorial <discrete-hopfield-network>`.

Before run all experiments, we need to setup ``seed`` parameter to make all results reproducible.
But you can test all outputs without it, jus to make sure that conclutions from your outputs are the same for most of all situations.

.. code-block:: python

    import random
    import numpy as np

    np.random.seed(0)
    random.seed(0)

Versions:

.. code-block:: python

    >>> import neupy
    >>> neupy.__version__
    '0.1.0'
    >>> import numpy
    >>> numpy.__version__
    '1.9.2'
    >>> import platform
    >>> platform.python_version()
    '3.4.3'

Data transformation
-------------------

Before build network that will save and recover password we must make transformations for the input and output data.
Lets assume that we are use only ASCII symbols.
We must make binary vector with a fixed length from a string.
Let's define a function that transform a string to the binary list.

.. code-block:: python

    import math
    import binascii
    from itertools import repeat

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

Function above takes 3 parameters.
First one is a string which we want to encode.
The second variable control of the same value for all rows.
Function adds spaces to the beginning of the line if the string length is less than the specified value in ``max_length``.
The third one control string encoding, but we will use only default ASCII encoding.

Let's test it with a word ``test``.

.. code-block:: python

    >>> str2bin("test", max_length=5)
    [0, 0, 1, 0, 0, 0, 0, 0, 0, ... ]
    >>> len(str2bin("test", max_length=5))
    40

ASCII use 8 bits per symbol and we set up 5 symbols per string, so our vector length always equal to 40.
As you can see first 8 symbols from output have form ``00100000`` which is space value from the ASCII table.

Now we must add another function that transform binary vector into the string.

.. code-block:: python

    def bin2str(array, encoding='ascii'):
        string_array = map(str, array)
        binary_string = ''.join(string_array)
        binary_string = '0b' + binary_string
        hex_string = bytearray(hex(int(binary_string, 2)), encoding)

        raw_text = binascii.unhexlify(hex_string[2:])
        return raw_text.lstrip().decode(encoding)

This function takes just two arguments.
First one is a binary vector.
The second one is the string encoding that we wouldn't update in this tutorial.

When we test it we get string ``test`` back.

.. code-block:: python

    >>> bin2str(str2bin("test", max_length=5))
    'test'

Pay attention, function removed all spaces at the beggining of the string before return it.

Save password into the network
------------------------------

Now we are ready to save the password into the network.
For this task we are going to define another function that create network and save password in it.
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

If you already read :ref:`Discrete Hopfield Network tutorial <discrete-hopfield-network>`, you must know that if we add only one vector into the network we will get it dublicated in whole matrix (sometimes with reversed signs).
To make it little bit secure we can add the noize into the network.
For this reason we define one additional parameter ``noize_level`` into the function.
We encode our password into the binary vector and save it into the ``data`` variable.
Next we using Binomial distribution generate random binary vectors where probability to get 1 in the vector is equal to 55%.
Parameter ``noize_level`` just control number of randomly generated binary vectors.

And finaly we define :network:`DiscreteHopfieldNetwork` instance.
We train the network with password binary vector and with all randomly generated binary vectors.
And that's it.
Function returns trained network for later usage.

But why do we use random binary vectors instead of the decoded random strings?
The problem is in the similarity between two vectors.
Let's check two approaches and compare them with a `Hamming distance <https://en.wikipedia.org/wiki/Hamming_distance>`_.
Before that we must define a function that compare distance between two vectors.

.. code-block:: python

    import string
    import random

    def hamming_distance(left, right):
        left, right = np.array(left), np.array(right)
        if left.shape != right.shape:
            raise ValueError("Shapes must be equal")
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


In addition we add the ``generate_password`` function that we will use for the tests.
Let's check Hamming distance between two randomly generate password vectors.

.. code-block:: python

    >>> hamming_distance(str2bin(generate_password(20, 20)),
    ...                  str2bin(generate_password(20, 20)))
    70

As we can see two randomly generated passwords are very similar to each other (approximetly 70% of bits are the same).
But If we compare randomly generated password and random binary vector we will see the difference.

.. code-block:: python

    >>> hamming_distance(str2bin(generate_password(20, 20)),
    ...                  np.random.binomial(1, 0.55, 240))
    122

Hamming distance is bigger than in the previous example.
Little bit more than 50% of the bits are different.
The bigger difference between random binary vector and string is improve probability to recover valid passowrd from the network.

Recover password from the network
---------------------------------

Now we are going to define the last function which will recover password from the network.

.. code-block:: python

    def recover_password(dhnet, broken_password):
        test = np.array(str2bin(broken_password))
        recovered_password = dhnet.predict(test)

        try:
            if recovered_password.ndim == 2:
                recovered_password = recovered_password[0, :]
            password = bin2str(recovered_password)

        except (UnicodeDecodeError, binascii.Error):
            # Panic mode
            password = generate_password()

        return password

As input function takes two parameters.
The first one is the network instance.
The second one is a broken password.
In function we can also see ``try ... except`` condition that fix problem if network return broken vector which we can't convert to the string.

Finnaly we can test it.

.. code-block:: python

    >>> my_password = "$My%Super^Secret*^&Passwd"
    >>> dhnet = save_password(my_password, noize_level=12)
    >>> recover_password(dhnet, "-My-Super-Secret---Passwd")
    '$My%Super^Secret*^&Passwd'
    >>> _ == my_password
    True
    >>>
    >>> recover_password(dhnet, "-My-Super")
    'y$!I}v^r(d`v7'
    >>>
    >>> recover_password(dhnet, "Invalid")
    'g9@wx\\/w8k;5P(N-9{A@U'
    >>>
    >>> recover_password(dhnet, "MySuperSecretPasswd")
    '$My%Super^Secret*^&Passwd'
    >>> _ == my_password
    True

Everithing looks fine.
But one problem sometimes exists.
Network can produce string that we didn't teach it.
This string can looks almost like the password, few replaced symbols.
Basicly each trained input vector create local minimum for the Discrete Hopfield Energy Function inside the network.
The problem is exists when network creates additional local minimum somewhere between input patterns.

Test it using Monte Carlo
-------------------------

Let's test it on a randomly generated passwords.
For this task we will run Monte Carlo experiment.

.. code-block:: python

    import pprint
    from operator import itemgetter
    from collections import OrderedDict

    def cutword(word, k, fromleft=False):
        if fromleft:
            return (word[-k:] if k != 0 else '').rjust(len(word))
        return (word[:k] if k != 0 else '').ljust(len(word))

    n_times = 10000
    cases = OrderedDict([
        ('exclude-one', (lambda x: x - 1)),
        ('exclude-quarter', (lambda x: 3 * x // 4)),
        ('exclude-half', (lambda x: x // 2)),
        ('just-one-symbol', (lambda x: 1)),
        ('empty-string', (lambda x: 0)),
    ])
    results = OrderedDict.fromkeys(cases.keys(), 0)

    for _ in range(n_times):
        real_password = generate_password(min_length=25, max_length=25)

        for casename, func in cases.items():
            n_letters = func(len(real_password))
            broken_password = cutword(real_password, k=n_letters,
                                      fromleft=True)

            dhnet = save_password(real_password, noize_level=11)
            recovered_password = recover_password(dhnet, broken_password)

            if recovered_password != real_password:
                results[casename] += 1

    print("Number of fails for each test case:")
    pprint.pprint(results)

Your output must be the same as the one below::

    Number of fails for each test case:
    {'exclude-one': 15,
     'exclude-quarter': 710,
     'exclude-half': 5663,
     'just-one-symbol': 9998,
     'empty-string': 10000}

On this test we catch two situation when the network recover password from an one symbol, which is not very good, but it really depence on the noize which we stored inside the network.
Sometimes it can recover password from an empty string, but the problem is similar to the one that we already catch.

Possible Problems
-----------------

There are few possible problems in the Discrete Hopfile Network.

1. Shifted words are harder to recover than the words with the missed symbols.

2. There already exists small probability to recover the password from the empty string.


Summary
-------

Despite some of the problems, network recovers password very good.
Monte Carlo experiment shows that the fewer symbols we know about the network less probability to recover it.
Even with a half of the known symbols we can recover password with probability less that 50%.


Download script
---------------

You can download and test a full script from the `github <https://github.com/itdxer/neuralpy/tree/master/examples/password_recovery.py>`_

It contains random ``seed`` initializations.
If you want test it in a random mode, just remove two lines with the random ``seed`` initializer from the script.

.. author:: default
.. categories:: none
.. tags:: memory, unsupervised
.. comments::
