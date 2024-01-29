import logging
from typing import Any

import torch

# from Huggingface
from tokenizers import Tokenizer

from mt_transformer.data_handler.data_loader import create_tokenizers_dataloaders
from mt_transformer.model.transformer_model import TransformerModel, get_transformer_model
from mt_transformer.trainer.transformer_validator import TransformerValidator
from mt_transformer.utils.tf_utils import get_proc_device


class TfInference:

    def __init__(self, config) -> None:
        self.__config = config

    def perform_inference(self) -> None:

        # get Cuda-GPU if available otherwise CPU
        device = get_proc_device()

        # get data loaders and tokenizers
        _, val_dataloader, tokenizer_src, tokenizer_tgt = \
            create_tokenizers_dataloaders(self.__config)

        # create transformer model
        transformer_model = get_transformer_model(
            self.__config,
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size()).to(device)

        # load trained state into model
        trained_model_epoch = self.__config.INFERENCE["trained_model_epoch"]
        logging.info("TfInference: load trained model at epoch %s",
                     f'{trained_model_epoch:03d}')

        state = torch.load(self.__config.get_saved_model_file_path(
            epoch=f"{trained_model_epoch:03d}"),
            map_location=torch.device(device))
        transformer_model.load_state_dict(state['model_state_dict'])

        # perform validation - translate
        logging.info('TfInference: perform validation')
        transformer_val = TransformerValidator()
        transformer_val.perform_validation(
            transformer_model, val_dataloader, tokenizer_src,
            tokenizer_tgt, self.__config.DATA['seq_len'], device,
            lambda msg: print(msg), 0, None, num_examples=10)

        logging.info('TfInference: perform translation')
        self.__perform_translation(
            transformer_model, tokenizer_src, tokenizer_tgt,
            self.__config.DATA['seq_len'], device)

    def __perform_translation(self,
                              transformer_model: TransformerModel,
                              tokenizer_src: Tokenizer,
                              tokenizer_tgt: Tokenizer,
                              seq_len: str,
                              device: str,
                              sentence: str = "I am a big fan of transformer models") -> Any:

        transformer_model.eval()
        with torch.no_grad():
            # pre compute the encoder output and reuse it for every generation step
            tokenized_sentence = tokenizer_src.encode(sentence)

            source = torch.cat([
                torch.tensor([tokenizer_src.token_to_id('[SOS]')],
                             dtype=torch.int64),
                torch.tensor(tokenized_sentence.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[EOS]')],
                             dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[PAD]')] *
                             (seq_len - len(tokenized_sentence.ids) - 2),
                             dtype=torch.int64)
            ], dim=0).to(device)

            source_mask = (source != tokenizer_src.token_to_id(
                '[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
            encoder_output = transformer_model.encode(source, source_mask)

            # initialize the decoder input with the sos token
            decoder_input = torch.empty(1, 1).fill_(
                tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

            # print the source sentence and the predicted translation
            print(f"{f'Source: ':>12}{sentence}")
            print(f"{f'Predicted: ':>12}", end='')

            # generate the translation word by word
            while decoder_input.size(1) < seq_len:

                # build decoder mask and decode
                decoder_mask = torch.triu(
                    torch.ones((1, decoder_input.size(
                        1), decoder_input.size(1))),
                    diagonal=1).type(torch.int).type_as(source_mask).to(device)

                decoded_output = transformer_model.decode(
                    encoder_output, source_mask, decoder_input,
                    decoder_mask)

                # project next token
                prob = transformer_model.project(decoded_output[:, -1])
                _, next_word = torch.max(prob, dim=1)

                decoder_input = torch.cat(
                    [decoder_input, torch.empty(1, 1).type_as(source).fill_(
                        next_word.item()).to(device)], dim=1)

                # print the translated word
                print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

                # stop if the end of sentence token is predicted
                if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                    break

        # convert ids to tokens
        return tokenizer_tgt.decode(decoder_input[0].tolist())
