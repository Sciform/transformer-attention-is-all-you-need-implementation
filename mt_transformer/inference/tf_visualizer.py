import logging
import pandas as pd

import altair 
import torch
from torch.utils.data import DataLoader

from mt_transformer.config.project_config import Config
from mt_transformer.data_handler.data_loader import create_tokenizers_dataloaders
from mt_transformer.model.greedy_decoder import GreedyDecoder
from mt_transformer.model.transformer_model import Transformer, get_transformer_model
from mt_transformer.trainer.transformer_validator import TransformerValidator
from mt_transformer.utils.tf_utils import get_proc_device


class TfVisualizer:

    def __init__(self, config) -> None:
        self.__config = config
        self.__device = None
        self.__val_dataloader = None
        self.__tokenizer_src = None
        self.__tokenizer_tgt = None
        self.__transformer_model = None


    def __get_data_and_model(self) -> None:
        
        # get Cuda-GPU if available otherwise CPU
        self.__device = get_proc_device()

        # get data loaders and tokenizers
        _, self.__val_dataloader, self.__tokenizer_src, self.__tokenizer_tgt = \
            create_tokenizers_dataloaders(self.__config)

        # create transformer model
        self.__transformer_model = get_transformer_model(self.__config, self.__tokenizer_src.get_vocab_size(), 
            self.__tokenizer_tgt.get_vocab_size()).to(self.__device)

        # load pretrained state into model
        state = torch.load(self.__config.get_saved_model_file_path(epoch=f"29"))
        self.__transformer_model.load_state_dict(state['model_state_dict'])
        
    
    def __load_next_batch(self):
        
        # Load a the next batch from the validation set
        batch = next(iter(self.__val_dataloader))
        encoder_input = batch["encoder_input"].to(self.__device)
        encoder_mask = batch["encoder_mask"].to(self.__device)
        decoder_input = batch["decoder_input"].to(self.__device)
        decoder_mask = batch["decoder_mask"].to(self.__device)

        encoder_input_tokens = [self.__tokenizer_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
        decoder_input_tokens = [self.__tokenizer_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

        # check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be >= 1 for validation"
        
        greedy_decoder = GreedyDecoder()
        tf_model_output = greedy_decoder.greedy_decode(self.__transformer_model, encoder_input, 
                                                       encoder_mask, self.__tokenizer_src,self.__tokenizer_tgt, 
                                                       self.__config.DATA['seq_len'], self.__device)
        
        return batch, encoder_input_tokens, decoder_input_tokens
    
    
def __mtx2df(self, m, max_row, max_col, row_tokens, col_tokens) -> pd.DataFrame:
    
    return pd.DataFrame(
        [
            (
                r_id,
                c_id,
                float(m[r_id, c_id]),
                "%.3d %s" % (r_id, row_tokens[r_id] if len(row_tokens) > r_id else "<blank>"),
                "%.3d %s" % (c_id, col_tokens[c_id] if len(col_tokens) > c_id else "<blank>"),
            )
            for r_id in range(m.shape[0])
            for c_id in range(m.shape[1])
            if r_id < max_row and c_id < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def __get_attn_map(self, attn_type: str, layer: int, head: int):
    
    if attn_type == "encoder":
        attn = self.__transformer_model.encoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "decoder":
        attn = self.__transformer_model.decoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "encoder-decoder":
        attn = self.__transformer_model.decoder.layers[layer].cross_attention_block.attention_scores
    return attn[0, head].data


def __attn_map(self, attn_type, layer, head, row_tokens, col_tokens, max_sentence_len):
    
    df = self.__mtx2df(
        self.__get_attn_map(attn_type, layer, head),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    
    return (
        altair.Chart(data=df)
        .mark_rect()
        .encode(
            x=altair.X("col_token", axis=altair.Axis(title="")),
            y=altair.Y("row_token", axis=altair.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.title(f"Layer {layer} Head {head}")
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )

def get_all_attention_maps(self, 
                           attn_type: str, 
                           layers: list[int], 
                           heads: list[int], 
                           row_tokens: list, 
                           col_tokens,    
                           max_sentence_len: int):
    
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(self.__attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(altair.hconcat(*rowCharts))
        
    return altair.vconcat(*charts)
        