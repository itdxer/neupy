.. _password-recovery:

Password recovery
=================

.. contents::

At this tutorial we are going to build a simple network that will recover password from a broken one.
If you don't familiar with a :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` algorithm, you can read :ref:`this tutorial <discrete-hopfield-network>`.

Before run all experiments, we need to setup ``seed`` parameter to make all results reproducible.
But you can test code without it.

.. code-block:: python

    import random
    import numpy as np

    np.random.seed(0)
    random.seed(0)

If you can't reproduce with your own version of Python and libraries you can install the same as I used:

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

Code works with a Python 2.7 as well.

Data transformation
-------------------

Before build the network that will save and recover the password we must make transformations for the input and output data.
Lets assume that we are use only ASCII symbols.
We must make binary vector with a fixed length from a string to make a vector with a constant length.
Let's define a function that transform a string to the binary list.

.. code-block:: python

    def str2bin(text, max_length=30):
        if len(text) > max_length:
            raise ValueError("Text can't contains more "
                             "than {} symbols".format(max_length))

        text = text.rjust(max_length)

        bits_list = []
        for symbol in text:
            bits = bin(ord(symbol))
            # Cut `0b` from the beggining and fill with zeros if they
            # are missed
            bits = bits[2:].zfill(8)
            bits_list.extend(map(int, bits))

        return list(bits_list)

Function above takes 2 parameters.
First one is a string which we want to encode.
The second attribute is set up a constant length for the input vector.
If length of the input string is less than the ``max_length`` value, then function adds spaces at the beginning of the string.

Let's test ``str2bin`` function.

.. code-block:: python

    >>> str2bin("test", max_length=5)
    [0, 0, 1, 0, 0, 0, 0, 0, 0, ... ]
    >>> len(str2bin("test", max_length=5))
    40

ASCII use 8 bits per symbol and we set up 5 symbols per string, so our vector length always equal to 40.
As you can see first 8 symbols from output have form ``00100000`` which is space value from the ASCII table.

Now we must add another function that makes inverse transformion from the binary vector to the string.

.. code-block:: python

    def chunker(sequence, size):
        for position in range(0, len(sequence), size):
            yield sequence[position:position + size]


    def bin2str(array):
        characters = []
        for binary_symbol_code in chunker(array, size=8):
            binary_symbol_str = ''.join(map(str, binary_symbol_code))
            character = chr(int(binary_symbol_str, base=2))
            characters.append(character)
        return ''.join(characters).lstrip()

If we test it we will get string ``test`` back.

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

If you are already read :ref:`Discrete Hopfield Network tutorial <discrete-hopfield-network>`, you must know that if we add only one vector into the network we will get it dublicated or with reversed signs in the whole matrix.
To make it little bit secure we can add the noize into the network.
For this reason we define one additional parameter ``noize_level`` into the function.

First of all we encode our password into the binary vector and save it into the ``data`` variable.
Next we using Binomial distribution generate random binary vectors with a 55% probability to get a 1 in the vector.
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
        password_len = random.randrange(min_length, max_length + 1)
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
    134

Hamming distance is bigger than in the previous example.
Little bit more than 55% of the bits are different.
The bigger difference between random binary vector the higher probability to recover valid passowrd from the network.

Recover password from the network
---------------------------------

Now we are going to define the last function which will recover password from the network.

.. code-block:: python

    def recover_password(dhnet, broken_password):
        test = np.array(str2bin(broken_password))
        recovered_password = dhnet.predict(test)

        if recovered_password.ndim == 2:
            recovered_password = recovered_password[0, :]

        return bin2str(recovered_password)

Function takes two parameters.
The first one is the network instance.
The second one is a broken password.

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
    '\x19`\xa0\x04Í\x14#ÛE2er\x1eÛe#2m4jV\x07PqsCwd'
    >>>
    >>> recover_password(dhnet, "Invalid")
    '\x02 \x1d`\x80$Ì\x1c#ÎE¢eò\x0eÛe§:/$ê\x04\x07@5sCu$'
    >>>
    >>> recover_password(dhnet, "MySuperSecretPasswd")
    '$My%Super^Secret*^&Passwd'
    >>> _ == my_password
    True

Everithing looks fine.
But one problem sometimes exists.
Network can produce string that we didn't teach it.
This string can looks almost like the password with few different symbols.
Basicly each trained input vector create local minimum inside of the Discrete Hopfield Network.
The problem is exists when network creates additional local minimum somewhere between input patterns.
We cann't defend from being hit into it.

Test it using Monte Carlo
-------------------------

Let's test it on a randomly generated passwords.
For this task we can use Monte Carlo experiment.

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


After sumbmission your output must be the same as the one below (if you make all step by step)::

    Number of fails for each test case:
    {'exclude-one': 11,
     'exclude-quarter': 729,
     'exclude-half': 5823,
     'just-one-symbol': 9998,
     'empty-string': 10000}

On this test we catch two situation when the network recover password from an one symbol, which is not very good.
It really depence on the noize which we stored inside the network.
Randomization can't give you a perfect result.
Sometimes it can recover password from an empty string, but it also rare situation.

Possible Problems
-----------------

There are few possible problems in the Discrete Hopfile Network.

1. Shifted words are harder to recover than the words with the missed symbols. Better to replace missed symbol with some other instead of remove it.

2. There already exists small probability to recover the password from the empty string.

3. Similar binary code representation for the different symbols is a big problem.
Some times you can have a situation when 2 symbols that are in binary form have difference in one bit. The first idea use a One Hot Encoder. But the problem with it is even bigger. For example we used one of the 94 symbols for the password. If we encode them for each symbol we will get vector with 93 zeros and just 1 active value. The problem that after recovery procedure we must always get a 1 active value which is very unlikely for the network.

Summary
-------

Despite some of the problems, network recovers password very good.
Monte Carlo experiment shows that the fewer symbols we know about the network less probability to recover it.
Even with a half of the known symbols we can recover password with probability less that 50%.

Download script
---------------

You can download and test a full script from the `github <https://github.com/itdxer/neupy/tree/master/examples/password_recovery.py>`_

It didn't contains random ``seed`` initializations, so you will get a different outputs after each run.

.. author:: default
.. categories:: none
.. tags:: memory, unsupervised
.. comments::
