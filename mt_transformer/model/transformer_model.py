from typing import Any
import torch
import torch.nn as nn
from mt_transformer.config.project_config import Config

from mt_transformer.model.layers import MultiHeadAttention, ResidualConnection
from mt_transformer.model.layers import LayerNormalization, FeedForwardBlock
from mt_transformer.model.layers import TokenEmbeddings, PositionalEncoding
from mt_transformer.model.layers import ProjectionLayer


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

        d_model = config.MODEL["d_model"]
        self.__src_seq_length = config.DATA["seq_len"]
        self.__tgt_seq_length = config.DATA["seq_len"]

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
