import math
from typing import Any

import torch
import torch.nn as nn


class TokenEmbeddings(nn.Module):
    """The token embeddings layer module holds a tensor that learns "d_model" 
    embeddings (= features) for every token in a fixed size dictionary.

    :param d_model: number of features per token
    :type d_model: int
    :param dictionary_size: length of dictionary
    :type dictionary_size: int
    """

    def __init__(self,
                 d_model: int,
                 dictionary_size: int) -> None:
        super().__init__()

        # number of features
        self.__d_model = d_model
        # embedding layer (dict_size x d_model)
        self.__embedding = nn.Embedding(dictionary_size, d_model)

    def forward(self, x: Any) -> Any:
        """ Compute embedding layer

        For every token in a token sequence of a batch of token sequences 
        :math:`x`, embeddings of dimension :math:`d_{\text{model}}` 
        are learned. A map :math:`x` with dim(num_batch, sequence_length) 
        to embeddings tensor with 
        dim(num_batch, sequence_length, d_model) is performed.
        
        In the original paper, the embedding tensor is multiplied by 
        :math:`\sqrt{d_{\text{model}}}` to scale the embeddings, but the reason for 
        this scaling is not explicitly explained.
        Some discussions on the topic can be found on 
        `Data Science Stack Exchange <https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod>`_ 
        and other related sources, where various considerations have been proposed 
        regarding the purpose of this scaling factor.
        
        It seems that the embedding matrix was originally initialized using a Gaussian 
        distribution with mean 0 and variance :math:`\frac{1}{d_{\text{model}}}`, i.e., 
        :math:`\mathcal{N}(0, fract{1}{d_{\text{model}}})`. 
        Therefore, the scaling factor :math:`\sqrt{d_{\text{model}}}` was applied to 
        bring the embeddings into a range closer to :math:`[-1,1]`, similar to the 
        positional encodings. The embedding layer from PyTorch is already initialized
        with the normal distribution :math:`\mathcal{N}(0, 1)`, 
        so the scaling factor is not necessary.

        :param x: batch of token sequences (tensor with dim(num_batch, sequence_length)) 
            for source and target text
        :type x: Any
        :return: embedding layer with learned "d_model" embeddings for every 
            token in the dictionary
        :rtype: Any

        """

        return self.__embedding(x) * math.sqrt(self.__d_model)


class PositionalEncoding(nn.Module):
    """ The positional encoding layer module holds a tensor with precomputed 
    positional values for every token in a sequence.

    :param d_model: number of features per token
    :type d_model: int
    :param seq_len: length of token sequence
    :type seq_len: int
    :param drop_out_prob: probability for drop out
    :type drop_out_prob: float

    """

    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 dropout_prob: float) -> None:
        super().__init__()

        self.__drop_out_layer = nn.Dropout(dropout_prob)

        # create a tensor of shape (seq_len, d_model) filled with zeros
        positional_encoding = torch.zeros(seq_len, d_model)

        # create a tensor of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # create a tensor of shape (d_model) for every embedding
        # the using exp and log leads to the same result but makes the computation numerically (better stability
        # https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ and
        # https://ai.stackexchange.com/questions/41670/why-use-exponential-and-log-in-positional-encoding-of-transformer
        denominator = torch.exp(torch.arange(0, d_model, 2).float() *
                                (-math.log(10000.0) / d_model))  # (d_model / 2)

        # compute sine-function for even indices
        # sin(position / (10000 ** (2i / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * denominator)
        # Compute cosine-function for odd indices
        # cos(position / (10000 ** (2i / d_model))
        positional_encoding[:, 1::2] = torch.cos(position * denominator)

        # Add a batch dimension to the positional encoding (1, seq_len, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)

        # register the positional encoding as a buffer - registering store the positional encoding with the model during
        # model save
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: Any) -> Any:
        """ Forwards the positional encodings

        Adds the positional encoding to the previous encoding tensor, which 
        usually contains the input encoding. The shape of the positional 
        encoding tensor is dim(batch, seq_len, d_model). 
        The method "requires_grad_(False)" tells the model
        that the positional encodings are precomputed and not learned
        A drop out is applied to the resulting tensor.

        :param x: input encoding tensor
        :type x: Any
        :return: Positional encoding with drop out applied to the input module
        :rtype: Any
        """

        # positional encoding is registered !!!
        x = x + (self.positional_encoding[:, :x.shape[1], :]). \
            requires_grad_(False)

        return self.__drop_out_layer(x)


class MultiHeadAttention(nn.Module):
    """ Layer module for multi-head attention

    :param d_model: number of embedding features
    :type d_model: int
    :param num_heads: number of attention heads
    :type num_heads: int
    :param dropout_prob: probability of drop out
    :type dropout_prob: float
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout_prob: float) -> None:
        super().__init__()

        self.__num_heads = num_heads  # number of heads
        # Make sure d_model is divisible by h
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        self.__d_k = d_model // num_heads  # Dimension of vector seen by each head
        # TODO why no bias ???
        self.__w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.__w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.__w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.__w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

        self.__dropout = nn.Dropout(dropout_prob)

    def __attention(self, query, key, value, mask):

        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask
        if mask is not None:
            # Write a very low value (approximating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        # (batch, h, seq_len, seq_len) # Apply softmax
        attention_scores = attention_scores.softmax(dim=-1)

        if self.__dropout is not None:
            attention_scores = self.__dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        # tuple attention_scores x values
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.__w_q(q)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.__w_k(k)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.__w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.__num_heads, self.__d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                       self.__num_heads, self.__d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.__num_heads, self.__d_k).transpose(1, 2)

        # Calculate attention
        x, _ = self.__attention(query, key, value, mask)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1,
                                                self.__num_heads * self.__d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.__w_o(x)


class LayerNormalization(nn.Module):
    """
    https://www.pinecone.io/learn/batch-layer-normalization/

    """

    def __init__(self,
                 eps: float = 10 ** -6) -> None:
        super().__init__()
        self.__eps = eps  # to avoid division by zero
        # to provide a scaling option for the model (check this)
        # alpha is a multiplicative learnable parameter
        self.__alpha = nn.Parameter(torch.ones(1))
        # bias is an additive learnable parameter
        self.__bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Append mean at the end and keep the original values
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero when std is very small
        return self.__alpha * (x - mean) / (std + self.__eps) + self.__bias


class ResidualConnection(nn.Module):

    """
    Add previous layer and normalized current layer

    Check first sublayer and the norm or vice versa ???

    """

    def __init__(self,
                 dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, new_layer):
        # is this correct ???
        # shouldn t it be previous x (sublayer) + attention_x
        # is sublayer the new layer
        return x + self.dropout(new_layer(self.norm(x)))


class FeedForwardBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float) -> None:
        super().__init__()
        self.__linear_layer_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.__dropout = nn.Dropout(dropout)
        self.__linear_layer_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.__linear_layer_2(self.__dropout(torch.relu(self.__linear_layer_1(x))))


class ProjectionLayer(nn.Module):

    def __init__(self,
                 d_model,
                 vocab_size) -> None:
        super().__init__()
        self.__projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.__projection(x), dim=-1)
