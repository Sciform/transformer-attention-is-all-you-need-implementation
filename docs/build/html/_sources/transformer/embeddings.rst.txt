

The Embeddings Layer
=================================

What are embeddings ?
---------------------------------



My take on scaling the embeddings
---------------------------------

The original paper by Vaswani as well the original implementation by Google
mention a scaling of the embedding by :math:`\sqrt{d_{model}},`
where :math:`d_{model}` refers to the number of features for each token.
However, neither in the paper nor in 

.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}

Or if you want to write inline you can use this:

:math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}`


