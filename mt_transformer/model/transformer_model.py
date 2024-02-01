
from typing import Any
import torch
import torch.nn as nn
from mt_transformer.config.project_config import Config

from mt_transformer.model.layers import MultiHeadAttention, ResidualConnection, LayerNormalization, FeedForwardBlock, TokenEmbeddings, PositionalEncoding, ProjectionLayer


class EncoderStackOld(nn.Module):

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
        :param src_mask: the source mask avoids that actual tokens interact with padding token \<PAD\>
        :return:
        """
        # multi head self attention, add & norm
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # feed forward, add & norm
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class EncoderOld(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        # why again a normalization
        return self.norm(x)


class EncoderStack(nn.Module):

    def __init__(self,
                 d_model: int,
                 h: int,
                 dropout_prob: float,
                 d_ff: int) -> None:
        super().__init__()
        self.__self_attention_block = MultiHeadAttention(
            d_model, h, dropout_prob)
        self.__feed_forward_block = FeedForwardBlock(
            d_model, d_ff, dropout_prob)
        self.__residual_connections = nn.ModuleList(
            [ResidualConnection(dropout_prob) for _ in range(2)])

    def forward(self, x, src_mask):
        """

        :param x:
        :param src_mask: the source mask avoids that actual tokens interact 
            with padding token \<PAD\>
        :return:
        """
        # multi head self attention, add & norm
        x = self.__residual_connections[0](
            x, lambda x: self.__self_attention_block(x, x, x, src_mask))
        # feed forward, add & norm
        x = self.__residual_connections[1](x, self.__feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 num_stacks: int,
                 h: int,
                 dropout_prob: float,
                 d_ff: int) -> None:
        super().__init__()

        self.__layer_normalization = LayerNormalization()
        self.__encoder_layers = self.__create_encoder_layers(
            d_model, num_stacks, h, dropout_prob, d_ff)

    def __create_encoder_layers(self,
                                d_model: int,
                                num_stacks: int,
                                h: int,
                                dropout_prob: float,
                                d_ff: int) -> nn.ModuleList:

        encoder_blocks = []
        for _ in range(num_stacks):
            encoder_block = EncoderStack(d_model, h, dropout_prob, d_ff)
            encoder_blocks.append(encoder_block)
            return nn.ModuleList(encoder_blocks)

    def forward(self, x, mask):
        for layer in self.__encoder_layers:
            x = layer(x, mask)

        # why again a normalization
        return self.__layer_normalization(x)


class DecoderStackOld(nn.Module):

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
                                         lambda x: self.cross_attention_block(x, encoder_output,
                                                                              encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class DecoderOld(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.__layers = layers
        self.__layer_normalization_layer = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.__layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.__layer_normalization_layer(x)


class DecoderStack(nn.Module):

    def __init__(self,
                 d_model: int,
                 h: int,
                 dropout_prob: float,
                 d_ff: int) -> None:
        super().__init__()
        self.__self_attention_block = MultiHeadAttention(
            d_model, h, dropout_prob)
        self.__cross_attention_block = MultiHeadAttention(
            d_model, h, dropout_prob)
        self.__feed_forward_block = FeedForwardBlock(
            d_model, d_ff, dropout_prob)
        self.__residual_connections = (
            nn.ModuleList([ResidualConnection(dropout_prob) for _ in range(3)]))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """

        :param x:
        :param encoder_output:
        :param src_mask:
        :param tgt_mask: to establish a causal model
        :return:
        """
        x = self.__residual_connections[0](
            x, lambda x: self.__self_attention_block(x, x, x, tgt_mask))
        x = self.__residual_connections[1](
            x, lambda x: self.__cross_attention_block(
                x, encoder_output, encoder_output, src_mask))
        x = self.__residual_connections[2](x, self.__feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 num_stacks: int,
                 h: int,
                 dropout_prob: float,
                 d_ff: int) -> None:
        super().__init__()

        self.__layer_normalization = LayerNormalization()
        self.__decoder_layers = self.__create_decoder_layers(
            d_model, num_stacks, h, dropout_prob, d_ff)

    def __create_decoder_layers(self,
                                d_model: int,
                                num_stacks: int,
                                h: int,
                                dropout_prob: float,
                                d_ff: int) -> nn.ModuleList:

        decoder_blocks = []
        for _ in range(num_stacks):
            decoder_block = DecoderStack(d_model, h, dropout_prob, d_ff)
            decoder_blocks.append(decoder_block)
        return nn.ModuleList(decoder_blocks)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.__decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.__layer_normalization(x)


class Transformer(nn.Module):

    def __init__(self,
                 encoder: EncoderOld,
                 decoder: DecoderOld,
                 src_embed: TokenEmbeddings,
                 tgt_embed: TokenEmbeddings,
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
               src_mask: torch.Tensor) -> EncoderOld:
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(self,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               tgt: torch.Tensor,
               tgt_mask: torch.Tensor) -> DecoderOld:
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x) -> ProjectionLayer:
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def create_transformer(src_vocab_size: int,
                       tgt_vocab_size: int,
                       src_seq_len: int,
                       tgt_seq_len: int,
                       d_model: int = 512,
                       num_stacks: int = 6,
                       h: int = 8,
                       dropout: float = 0.1,
                       d_ff: int = 2048) -> Transformer:

    # Create the embedding layers
    src_embed = TokenEmbeddings(d_model, src_vocab_size)
    tgt_embed = TokenEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder stacks
    encoder_blocks = []
    for _ in range(num_stacks):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderStackOld(
            encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder stacks
    decoder_blocks = []
    for _ in range(num_stacks):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderStackOld(decoder_self_attention_block,
                                        decoder_cross_attention_block,
                                        feed_forward_block,
                                        dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = EncoderOld(nn.ModuleList(encoder_blocks))
    decoder = DecoderOld(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos,
                              tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


"""
def get_transformer_model(config, vocab_src_len, vocab_tgt_len):

    model = create_transformer(vocab_src_len, vocab_tgt_len, config.DATA["seq_len"], config.DATA['seq_len'],
                               d_model=config.MODEL['d_model'])
    return model

"""


class TransformerModel(nn.Module):

    def __init__(self,
                 config: Config,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 num_stacks: int = 6,
                 h: int = 8,
                 dropout_prob: float = 0.1,
                 d_ff: int = 2048) -> None:
        super().__init__()

        d_model = config['d_model']
        self.__src_seq_length = config["seq_len"]
        self.__tgt_seq_length = config["seq_len"]

        # Create the embedding layers
        self.__src_embed = TokenEmbeddings(d_model, src_vocab_size)
        self.__tgt_embed = TokenEmbeddings(d_model, tgt_vocab_size)

        # Create the positional encoding layers
        self.__src_pos = PositionalEncoding(d_model, self.__src_seq_length,
                                            dropout_prob)
        self.__tgt_pos = PositionalEncoding(d_model, self.__tgt_seq_length,
                                            dropout_prob)

        # Create the encoder and decoder
        self.__encoder = Encoder(d_model, num_stacks, h, dropout_prob, d_ff)
        self.__decoder = Decoder(d_model, num_stacks, h, dropout_prob, d_ff)

        # Create projection layer
        self.__projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        # Initialize parameters
        self.__init_parameters()

    def __init_parameters(self):

        # Initialize the parameters of the transformer
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self,
               src: torch.Tensor,
               src_mask: torch.Tensor) -> Any:
        # (batch, seq_len, d_model)
        src = self.__src_embed(src)
        src = self.__src_pos(src)

        return self.__encoder(src, src_mask)

    def decode(self,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               tgt: torch.Tensor,
               tgt_mask: torch.Tensor) -> Any:

        # (batch, seq_len, d_model)
        tgt = self.__tgt_embed(tgt)
        tgt = self.__tgt_pos(tgt)

        return self.__decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x) -> ProjectionLayer:

        # (batch, seq_len, vocab_size)
        return self.__projection_layer(x)
