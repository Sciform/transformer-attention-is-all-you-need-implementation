import logging

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from tqdm import tqdm

from mt_transformer.config.project_config import Config
from mt_transformer.data_handler.data_loader import create_tokenizers_dataloaders
from mt_transformer.model.transformer_model import TransformerModel
from mt_transformer.trainer.transformer_validator import TransformerValidator
from mt_transformer.utils.tf_utils import get_proc_device


class TransformerTrainer:
    """ 
    The class handles the training flow of the MT_Transformer

    :param config: configuration
    :type config: Config
    """

    def __init__(self, config: Config) -> None:
        self.__config = config

    def __get_last_trained_epoch(self,
                                 transformer_model: TransformerModel,
                                 optimizer: Optimizer,
                                 device: str) -> [int, int]:
        """Get the last trained epoch of the model and update the optimizer

        :param transformer_model: transformer model
        :type transformer_model: Transformer
        :param optimizer: PyTorch optimizer
        :type optimizer: Optimizer
        :param device: device type
        :type device: str
        :return: epoch to be started with, global step of state
        :rtype: [int, int]
        """
        start_epoch = 0
        global_step = 0

        if self.__config.MODEL['pretrained_model_epoch'] is not None:

            last_epoch = self.__config.MODEL['pretrained_model_epoch']
            model_filename = self.__config.get_saved_model_file_path(
                f"{last_epoch:03d}")
            logging.info("Load pretrained model %s", model_filename)

            state = torch.load(model_filename,
                               map_location=torch.device(device))
            transformer_model.load_state_dict(state['model_state_dict'])

            start_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']

        return start_epoch, global_step

    def perform_training(self):
        """Perform model training

        """
        logging.info("Trainer: prepare transformer training!")

        # get Cuda-GPU if available otherwise CPU
        device = get_proc_device()

        ###
        # Data
        ###

        # get data loaders and tokenizers
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = \
            create_tokenizers_dataloaders(self.__config)

        ###
        # Model
        ###

        # create transformer model
        transformer_model = TransformerModel(
            self.__config, tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size()).to(device)

        # create Adam optimizer
        optimizer = torch.optim.Adam(
            transformer_model.parameters(),
            lr=self.__config.MODEL['learning_rate'],
            eps=self.__config.MODEL['eps'])

        # create loss function
        ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),
                                      label_smoothing=0.1).to(device)

        # load saved model if available
        start_epoch, global_step = self.__get_last_trained_epoch(
            transformer_model, optimizer, device)

        # get validator
        trans_val = TransformerValidator()

        # get tensorboard writer
        writer = SummaryWriter(self.__config.get_experiments_file_path())

        ###
        # Perform training
        ###
        logging.info("Trainer: start transformer training!")
        for epoch in range(start_epoch, self.__config.MODEL['num_epochs']):

            logging.info("Trainer: start transformer training of epoch %s!",
                         epoch)

            torch.cuda.empty_cache()
            transformer_model.train()

            batch_iterator = tqdm(train_dataloader,
                                  desc=f"Processing epoch {epoch:02d}")

            for batch in batch_iterator:
                # (b, seq_len)
                encoder_input = batch['encoder_input'].to(device)
                # (B, seq_len)
                decoder_input = batch['decoder_input'].to(device)
                # (B, 1, 1, seq_len)
                encoder_mask = batch['encoder_mask'].to(device)
                # (B, 1, seq_len, seq_len)
                decoder_mask = batch['decoder_mask'].to(device)

                # Encode the source data (B, seq_len, d_model)
                encoder_output = transformer_model.encode(
                    encoder_input, encoder_mask)
                # Decode the encoded source (B, seq_len, d_model)
                decoder_output = transformer_model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask)
                # Create transformer output (B, seq_len, vocab_size)
                model_output = transformer_model.project(decoder_output)

                # Compare the output with the label (B, seq_len)
                label = batch['label'].to(device)

                # Compute the loss using a simple cross entropy (input, target)
                # pylint: disable=not-callable
                loss = ce_loss(model_output.view(-1, tokenizer_tgt.get_vocab_size()),
                               label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss in Tensorboard
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                # Backpropagate the loss
                loss.backward()

                # update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            # perform validation at the end of every epoch
            trans_val.perform_validation(
                transformer_model, val_dataloader, tokenizer_tgt,
                self.__config.DATA['seq_len'], device,
                lambda msg: batch_iterator.write(msg), global_step, writer)

            # save the model at the end of every epoch
            saved_model_filepath = self.__config.get_saved_model_file_path(
                f"{epoch:03d}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step}, saved_model_filepath)
