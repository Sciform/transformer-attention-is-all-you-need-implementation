

The Embeddings Layer
=================================

What are embeddings ?
---------------------------------

Embeddings are vector representations of input tokens that capture their 
semantic meaning. Since Transformers do not process raw text, 
each token (word, subword, or character) 
is mapped to a dense numerical vector before being processed by the model.


My take on scaling the embeddings
---------------------------------

The original paper by Vaswani as well the original implementation by Google
mention a scaling of the embedding by :math:`\sqrt{d_{model}},`
where :math:`d_{model}` refers to the number of features for each token.
However, neither in the paper nor in 

In the original paper by Vaswani, the embedding tensor is multiplied by 
:math:`\sqrt{d_{\text{model}}}` to scale the embeddings, 
where :math:`d_{\text{model}}` is the number of features for each token.

However, the reason for 
this scaling is not explicitly explained.
Some discussions on the topic can be found on 
`Data Science Stack Exchange <https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod>`_ 
and other related sources such as 
`Tensor2tensor Issues <https://github.com/tensorflow/tensor2tensor/issues/1718>`, 
where various considerations have been proposed 
regarding the purpose of this scaling factor.

It seems that the embedding matrix was originally initialized using a Gaussian 
distribution with mean 0 and variance :math:`\frac{1}{d_{\text{model}}}`, i.e., 
:math:`\mathcal{N}(0, fract{1}{d_{\text{model}}})`. 
Therefore, the scaling factor :math:`\sqrt{d_{\text{model}}}` was applied to 
bring the embeddings into a range closer to :math:`[-1,1]`, similar to the 
positional encodings. 

The embedding layer from PyTorch is already initialized
with the normal distribution :math:`\mathcal{N}(0, 1)`, 
so the scaling factor is not necessary.


