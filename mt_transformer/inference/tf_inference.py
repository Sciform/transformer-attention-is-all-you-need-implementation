import logging

import torch

from mt_transformer.config.project_config import Config
from mt_transformer.data_handler.data_loader import create_tokenizers_dataloaders
from mt_transformer.model.transformer_model import get_transformer_model
from mt_transformer.trainer.transformer_validator import TransformerValidator
from mt_transformer.utils.tf_utils import get_proc_device


class TfInference:


    def __init__(self, config) -> None:
        self.__config = config


    def perform_inference(self) -> None:
        
        # get Cuda-GPU if available otherwise CPU
        device = get_proc_device()

        # get data loaders and tokenizers
        _, val_dataloader, tokenizer_src, tokenizer_tgt = create_tokenizers_dataloaders(self.__config)

        # create transformer model
        transformer_model = get_transformer_model(self.__config, tokenizer_src.get_vocab_size(), 
            tokenizer_tgt.get_vocab_size()).to(device)
        
        print("load model and state")
        print(device)

        # load trained state into model
        trained_model_epoch = self.__config.INFERENCE["trained_model_epoch"]
        state = torch.load(self.__config.get_saved_model_file_path(epoch=f"{trained_model_epoch:03d}"))
        transformer_model.load_state_dict(state['model_state_dict'], map_location=torch.device(device))
        
        print("perform validation resp")

        # perform validation - translate
        transformer_val = TransformerValidator()
        transformer_val.perform_validation(transformer_model, val_dataloader, tokenizer_src, 
                                           tokenizer_tgt, self.__config.DATA['seq_len'], device, 
                                           lambda msg: print(msg), 0, None, num_examples=10)
             