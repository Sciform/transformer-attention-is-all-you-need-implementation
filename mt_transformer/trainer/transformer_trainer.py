
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mt_transformer.data_handler.data_loader import create_tokenizers_dataloaders
from mt_transformer.model.transformer_model import get_transformer_model
from mt_transformer.trainer.transformer_validator import TransformerValidator
from mt_transformer.utils.tf_utils import get_proc_device


class TransformerTrainer:
    
    def __init__(self, config) -> None:
        self.__config = config
    
    def __get_preload_model_setup(self, transformer_model, optimizer):
        
        initial_epoch = 0
        global_step = 0
        
        if self.__config.MODEL['preload'] is not None: 

            model_filename = self.__config.get_saved_model_file_path(self.__config.MODEL['preload'])
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            transformer_model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
            
        return initial_epoch, global_step
            

    def perform_training(self):

        print("Start transformer!")

        # get Cuda-GPU if available otherwise CPU
        device = get_proc_device()
        
        ###
        ### Data
        ###

        # get data loaders and tokenizers
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = create_tokenizers_dataloaders(self.__config)

        ###
        ### Model
        ###
        
        # create transformer model
        transformer_model = get_transformer_model(self.__config, tokenizer_src.get_vocab_size(), 
            tokenizer_tgt.get_vocab_size()).to(device)
        
        # create Adam optimizer
        optimizer = torch.optim.Adam(transformer_model.parameters(), lr=self.__config.MODEL['lr'], eps=1e-9)
        
        # create loss function
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
        
        # load saved model if available
        initial_epoch, global_step = self.__get_initial_model_setup(self.__config, transformer_model, optimizer)

        # get validator
        trans_val = TransformerValidator()

        # get tensorboard writer
        writer = SummaryWriter(self.__config.get_experiments_file_path())
        
        ###
        ### Perform training
        ###

        print("Perform training")
        for epoch in range(initial_epoch, self.__config.MODEL['num_epochs']):

            torch.cuda.empty_cache()
            transformer_model.train()

            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = transformer_model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
                decoder_output = transformer_model.decode(encoder_output, encoder_mask, decoder_input,
                                              decoder_mask)  # (B, seq_len, d_model)
                proj_output = transformer_model.project(decoder_output)  # (B, seq_len, vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device)  # (B, seq_len)

                # Compute the loss using a simple cross entropy
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                # backpropagate the loss
                loss.backward()

                # update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            # perform validation at the end of every epoch
            trans_val.perform_validation(transformer_model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                                         self.__config.DATA['seq_len'], device, lambda msg: batch_iterator.write(msg), 
                                         global_step, writer)

            # save the model at the end of every epoch
            saved_model_filepath = self.__config.get_saved_model_file_path(f"{epoch:03d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step }, saved_model_filepath)
            