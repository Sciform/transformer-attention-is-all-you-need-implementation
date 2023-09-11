import math

import torch
import torch.nn as nn


class TextEmbeddings(nn.Module):
    """
    The text embeddings modul is a tensor that stores
    embeddings of a fixed size dictionary.

    This embedding module is used to store word embeddings for every token
    index in the dictionary.

    """

    def __init__(self, d_model: int, dictionary_size: int) -> None:
        super().__init__()
        self.__d_model = d_model
        self.__dictionary_size = dictionary_size
        self.__embedding = nn.Embedding(dictionary_size, d_model)

    def forward(self, x):
        """
        What is x - map between token and dictionary index ?
        Num_batch contains a number of token sequences.
        (num_batch, sequence_length) --> (num_batch, sequence_length, d_model)
        Multiply by sqrt(d_model) to scale the embeddings according to the paper

        :param x:
        :return:
        """
        return self.__embedding(x) * math.sqrt(self.__d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, drop_out: float) -> None:
        super().__init__()
        self.__d_model = d_model
        self.__seq_len = seq_len
        self.__drop_out_layer = nn.Dropout(drop_out)

        # Create a tensor of shape (seq_len, d_model) filled with zeros
        positional_encoding = torch.zeros(seq_len, d_model)
        # Create a tensor of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a tensor of shape (d_model) for every embedding
        # the using exp and log leads to the same result but makes the computation numerically
        # more stable
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)

        # Compute sine-function for even indices
        positional_encoding[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        # Compute cosine-function for odd indices
        positional_encoding[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        positional_encoding = positional_encoding.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):

        """
        Adds the positional encoding to the previous encoding tensor, which usually contains
        the input encoding. The shape of the positional encoding tensor is
        dim(batch, seq_len, d_model). The method "requires_grad_(False)" tells the model
        that the positional encodings are precomputed and not learned
        A drop out is applied to the resulting tensor.

        :param x: input encoding tensor
        :return: sum of input encoding and positional encoding with drop out
        """
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)

        return self.__drop_out_layer(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.num_heads = num_heads  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        self.d_k = d_model // num_heads  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask
        if mask is not None:
            # Write a very low value (approximating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        # tuple attention_scores x values
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):

        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # Calculate attention
        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class LayerNormalization(nn.Module):
    """
    https://www.pinecone.io/learn/batch-layer-normalization/

    """

    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps # to avoid division by zero
        # to provide a scaling option for the model (check this)
        self.alpha = nn.Parameter(torch.ones(1))  # alpha is a multiplicative learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))  # bias is an additive learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Append mean at the end and keep the original values
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):

    """
    Add previous layer and normalized current layer

    Check first sublayer and the norm or vice versa ???

    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, new_layer):
        # is this correct ???
        # shouldn t it be previous x (sublayer) + attention_x
        # is sublayer the new layer
        return x + self.dropout(new_layer(self.norm(x)))


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.__linear_layer_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.__dropout = nn.Dropout(dropout)
        self.__linear_layer_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.__linear_layer_2(self.__dropout(torch.relu(self.__linear_layer_1(x))))


class EncoderStack(nn.Module):

    def __init__(self,
                 self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """

        :param x:
        :param src_mask: the source mask avoids that actual tokens interact with padding token <PAD>
        :return:
        """
        # multi head self attention, add & norm
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # feed forward, add & norm
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        # why again a normalization
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)


class DecoderStack(nn.Module):

    def __init__(self,
                 self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = (
            nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """

        :param x:
        :param encoder_output:
        :param src_mask:
        :param tgt_mask: to establish a causal model
        :return:
        """
        x = self.residual_connections[0](x,
            lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x,
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: TextEmbeddings,
                 tgt_embed: TextEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self,
               src: torch.Tensor,
               src_mask: torch.Tensor) -> Encoder:
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(self,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               tgt: torch.Tensor,
               tgt_mask: torch.Tensor) -> Decoder:
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x) -> ProjectionLayer:
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      num_stacks: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:

    # Create the embedding layers
    src_embed = TextEmbeddings(d_model, src_vocab_size)
    tgt_embed = TextEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder stacks
    encoder_blocks = []
    for _ in range(num_stacks):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderStack(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder stacks
    decoder_blocks = []
    for _ in range(num_stacks):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderStack(decoder_self_attention_block,
                                     decoder_cross_attention_block,
                                     feed_forward_block,
                                     dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

