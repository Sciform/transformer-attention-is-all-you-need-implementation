
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.model_config import get_weights_file_path
from src.data.data_loader import create_tokenizers_dataloaders
from src.model.transformer_model import get_model
from trainer.transformer_validator import TransformerValidator


class TransformerTrainer:

    def perform_training(self, config):

        print("Start transformer!")

        # get GPU if available otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # generate folder for weights folder if it does not exist yet
        Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

        # get data loaders and tokenizers
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = create_tokenizers_dataloaders(config)

        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

        # get tensorboard writer
        writer = SummaryWriter(config['experiment_name'])

        # specify Adom optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

        trans_val = TransformerValidator()

        print("Configure model")
        # If the user specified a model to preload before training, load it
        initial_epoch = 0
        global_step = 0
        if config['preload']:
            model_filename = get_weights_file_path(config, config['preload'])
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']

        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

        print("Perform training")
        for epoch in range(initial_epoch, config['num_epochs']):

            torch.cuda.empty_cache()
            model.train()

            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                              decoder_mask)  # (B, seq_len, d_model)
                proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

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
            trans_val.perform_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                                        lambda msg: batch_iterator.write(msg), global_step, writer)

            # save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


