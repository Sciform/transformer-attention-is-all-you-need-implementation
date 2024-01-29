from typing import Any

import torch
from tokenizers import Tokenizer
from datasets import Dataset as HfDataSet

from mt_transformer.data_handler.masks import causal_mask


class TwoLanguagesDataset(torch.utils.data.Dataset):

    def __init__(self,
                 ds_raw: HfDataSet,
                 tokenizer_src: Tokenizer,
                 tokenizer_tgt: Tokenizer,
                 src_lang: str,
                 tgt_lang: str,
                 seq_len: int) -> None:
        super().__init__()

        self.__seq_len = seq_len
        self.__ds = ds_raw
        self.__tokenizer_src = tokenizer_src
        self.__tokenizer_tgt = tokenizer_tgt
        self.__src_lang = src_lang
        self.__tgt_lang = tgt_lang

        self.__sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")],
                                        dtype=torch.int64)
        self.__eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")],
                                        dtype=torch.int64)
        self.__pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")],
                                        dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.__ds)

    def __getitem__(self,
                    idx: int) -> dict[str, Any]:

        src_target_pair = self.__ds[idx]
        src_text = src_target_pair['translation'][self.__src_lang]
        tgt_text = src_target_pair['translation'][self.__tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.__tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.__tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.__seq_len - len(enc_input_tokens) - 2
        # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.__seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative.
        # If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.__sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.__eos_token,
                torch.tensor([self.__pad_token] * enc_num_padding_tokens,
                             dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.__sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.__pad_token] * dec_num_padding_tokens,
                             dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.__eos_token,
                torch.tensor([self.__pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.__seq_len
        assert decoder_input.size(0) == self.__seq_len
        assert label.size(0) == self.__seq_len

        return {
            # (seq_len)
            "encoder_input": encoder_input,
            # (seq_len)
            "decoder_input": decoder_input,
            # (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.__pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.__pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,
            # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
