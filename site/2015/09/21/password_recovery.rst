.. _password-recovery:

Password recovery
=================

.. raw:: html

    <div class="short-description">
        <p>
        Discrete hopfiled networks can be used to solve wide variety of problems. In this article, we try to use this type of network in order to memorizes user's password and then we try reconstruct it from partially corrupted version of this password.
        </p>
        <br clear="right">
    </div>

.. contents::

In this article we are going to build a simple neural network that will recover password from a broken one.
If you aren't familiar with a :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` algorithm, you can read :ref:`this article <discrete-hopfield-network>`.

Before running all experiments, we need to set up ``seed`` parameter to make all results reproducible.
But you can test code without it.

.. code-block:: python

    from neupy import utils
    utils.reproducible()

If you can't reproduce with your version of Python or libraries you can install those ones that were used in this article:

.. code-block:: python

    >>> import neupy
    >>> neupy.__version__
    '0.3.0'
    >>> import numpy
    >>> numpy.__version__
    '1.9.2'
    >>> import platform
    >>> platform.python_version()
    '3.4.3'

Code works with a Python 2.7 as well.

Data transformation
-------------------

Before building the network that will save and recover passwords, we should make transformations for input and output data.
But it wouldn't be enough just to encode it, we should set up a constant length for an input string to make sure that strings will have the same length
Also we should define what string encoding we will use.
For simplicity we will use only ASCII symbols.
So, let's define a function that transforms a string into a binary list.

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

Our function takes 2 parameters.
First one is the string that we want to encode.
And second attribute is setting up a constant length for input vector.
If length of the input string is less than ``max_length`` value, then function fills spaces at the beginning of the string.

Let's check ``str2bin`` function output.

.. code-block:: python

    >>> str2bin("test", max_length=5)
    [0, 0, 1, 0, 0, 0, 0, 0, 0, ... ]
    >>> len(str2bin("test", max_length=5))
    40

ASCII encoding uses 8 bits per symbol and we set up 5 symbols per string, so our vector length equals to 40.
From the first output, as you can see, first 8 symbols are equal to ``00100000``, that is a space value from the ASCII table.

After preforming recovery procedure we will always be getting a binary list.
So before we begin to store data in neural network, we should define another function that transforms a binary list back into a string (which is basically inversed operation to the previous function).

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

If we test this function we will get word ``test`` back.

.. code-block:: python

    >>> bin2str(str2bin("test", max_length=5))
    'test'

Pay attention! Function has removed all spaces at the beggining of the string before bringing them back.
We assume that password won't contain space at the beggining.

Saving password into the network
--------------------------------

Now we are ready to save the password into the network.
For this task we are going to define another function that create network and save password inside of it.
Let's define this function and later we will look at it step by step.

.. code-block:: python


    import numpy as np
    from neupy import algorithms

    def save_password(real_password, noise_level=5):
        if noise_level < 1:
            raise ValueError("`noise_level` must be equal or greater than 1.")

        binary_password = str2bin(real_password)
        bin_password_len = len(binary_password)

        data = [binary_password]

        for _ in range(noise_level):
            # The farther from the 0.5 value the less likely
            # password recovery
            noise = np.random.binomial(1, 0.55, bin_password_len)
            data.append(noise)

        dhnet = algorithms.DiscreteHopfieldNetwork(mode='sync')
        dhnet.train(np.array(data))

        return dhnet

If you have already read :ref:`Discrete Hopfield Network article <discrete-hopfield-network>`, you should know that if we add only one vector into the network we will get it dublicated or with reversed signs through the whole matrix.
To make it a little bit secure we can add some noise into the network.
For this reason we introduce one additional parameter ``noise_level`` into the function.
This parameter controls number of randomly generated binary vectors.
With each iteration using Binomial distribution we generate random binary vector with 55% probability of getting 1 in `noise` vector.
And then we put all the noise vectors and transformed password into one matrix.
And finaly we save all data in the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.

And that's it.
Function returns trained network for a later usage.

But why do we use random binary vectors instead of the decoded random strings?
The problem is in the similarity between two vectors.
Let's check two approaches and compare them with a `Hamming distance <https://en.wikipedia.org/wiki/Hamming_distance>`_.
But before starting we should define a function that measures distance between two vectors.

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


In addition you can see the ``generate_password`` function that we will use for tests.
Let's check Hamming distance between two randomly generate password vectors.

.. code-block:: python

    >>> hamming_distance(str2bin(generate_password(20, 20)),
    ...                  str2bin(generate_password(20, 20)))
    70

As we can see two randomly generated passwords are very similar to each other (approximetly 70% (:math:`100 * (240 - 70) / 240`) of bits are the same).
But If we compare randomly generated password to random binary vector we will see the difference.

.. code-block:: python

    >>> hamming_distance(str2bin(generate_password(20, 20)),
    ...                  np.random.binomial(1, 0.55, 240))
    134

Hamming distance is bigger than in the previous example.
A little bit more than 55% of the bits are different.

The greater the difference between them the easier recovery procedure for the input vectors patterns from the network.
For this reason we use randomly generated binary vector instead of random password.

Of course it's better to save not randomly generated noise vectors but randomly generated passwords converted into binary vectors, cuz if you use wrong input pattern randomly generated password might be recovered instead of the correct one.

Recovering password from the network
------------------------------------

Now we are going to define the last function which will recover a password from the network.

.. code-block:: python

    def recover_password(dhnet, broken_password):
        test = np.array(str2bin(broken_password))
        recovered_password = dhnet.predict(test)

        if recovered_password.ndim == 2:
            recovered_password = recovered_password[0, :]

        return bin2str(recovered_password)

Function takes two parameters.
The first one is network example from which function will recover a password from a broken one.
And the second parameter is a broken password.

Finnaly we can test password recovery from the network.

.. code-block:: python

    >>> my_password = "$My%Super^Secret*^&Passwd"
    >>> dhnet = save_password(my_password, noise_level=12)
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
After multiple times code running you can rarely find a problem.
Network can produce a string which wasn't taught.
This string can look almost like a password with a few different symbols.
The problem appears when network creates additional local minimum somewhere between input patterns.
We can't prevent it from running into the local minimum.
For more information about this problem you can check :ref:`article about Discrete Hopfield Network <discrete-hopfield-network>`.

Test it using Monte Carlo
-------------------------

Let's test our solution with randomly generated passwords.
For this task we can use Monte Carlo experiment.
At each step we create random password and try to recover it from a broken password.

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

            dhnet = save_password(real_password, noise_level=11)
            recovered_password = recover_password(dhnet, broken_password)

            if recovered_password != real_password:
                results[casename] += 1

    print("Number of fails for each test case:")
    pprint.pprint(results)

After sumbmission your output should look the same as the one below (if you followed everything step by step)::

    Number of fails for each test case:
    {'exclude-one': 11,
     'exclude-quarter': 729,
     'exclude-half': 5823,
     'just-one-symbol': 9998,
     'empty-string': 10000}

At this test we catch two situations when the network recovers the password from one symbol, which is not very good.
It really depends on the noise which we stored inside the network.
Randomization can't give you perfect results.
Sometimes it can recover a password from an empty string, but such situation is also very rare.

In the last test, on each iteration we cut password from the left side and filled other parts with spaces.
Let's test another approach.
Let's cut a password from the right side and see what we'll get::

    Number of fails for each test case:
    {'exclude-one': 17,
     'exclude-quarter': 705,
     'exclude-half': 5815,
     'just-one-symbol': 9995,
     'empty-string': 10000}

Results look similar to the previous test.

Another interesting test can take place if you randomly replace some symbols with spaces::

    Number of fails for each test case:
    {'exclude-one': 14,
     'exclude-quarter': 749,
     'exclude-half': 5760,
     'just-one-symbol': 9998,
     'empty-string': 10000}

The result is very similar to the previous two.

And finally, instead of replacing symbols with spaces we can remove symbols without any replacements.
Results do not look good::

    Number of fails for each test case:
    {'exclude-one': 3897,
     'exclude-quarter': 9464,
     'exclude-half': 9943,
     'just-one-symbol': 9998,
     'empty-string': 9998}

I guess in first case (``exclude-one``) we just got lucky and after eliminating one symbol from the end didn't shift most of the symbols.
So removing symbols is not a very good idea.

All functions that you need for experiments you can find at the `github <https://github.com/itdxer/neupy/tree/master/examples/memory/password_recovery.py>`_.

Possible problems
-----------------

There are a few possible problems in the Discrete Hopfile Network.

1. As we saw from the last experiments, shifted passwords are harder to recover than the passwords with missed symbols. It's better to replace missed symbols with some other things.

2. There already exists small probability for recovering passwords from empty strings.

3. Similar binary code representation for different symbols is a big problem. Sometimes you can have a situation where two symbols in binary code represantation are different just by one bit. The first solution is to use a One Hot Encoder. But it can give us even more problems. For example, we used symbols from list of 94 symbols for the password. If we encode each symbol we will get a vector with 93 zeros and just one active value. The problem is that after the recovery procedure we should always get 1 active value, but this situation is very unlikely to happen.

Summary
-------

Despite some problems, network recovers passwords very well.
Monte Carlo experiment shows that the fewer symbols we know the less is probability for recovering them correctly.

Even this simple network can be a powerful tool if you know its limitations.

Download script
---------------

You can download and test a full script from the `github repository <https://github.com/itdxer/neupy/tree/master/examples/memory/password_recovery.py>`_.

It doesn't contain a fixed ``utils.reproducible`` function, so you will get different outputs after each run.

.. author:: default
.. categories:: none
.. tags:: memory, unsupervised, discrete hopfield network
.. comments::
