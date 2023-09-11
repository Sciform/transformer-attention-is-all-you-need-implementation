import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from src.config.model_config import get_weights_file_path
from src.data.two_language_data_set import causal_mask
from src.data.data_loader import create_language_datasets
from src.model.transformer_model import get_model


class TransformerTrainer:

    def __greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encode(source, source_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break

            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

            # calculate output
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)

    def __run_validation(self, model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg,
                       global_step,
                       writer, num_examples=2):
        model.eval()
        count = 0

        source_texts = []
        expected = []
        predicted = []

        try:
            # get the console window width
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                console_width = int(console_width)
        except:
            # If we can't get the console width, use 80 as default
            console_width = 80

        with torch.no_grad():
            for batch in validation_ds:
                count += 1
                encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
                encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

                # check that the batch size is 1
                assert encoder_input.size(
                    0) == 1, "Batch size must be 1 for validation"

                model_out = self.__greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt,
                                               max_len,
                                               device)

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)

                # Print the source, target and model output
                print_msg('-' * console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

                if count == num_examples:
                    print_msg('-' * console_width)
                    break

        if writer:
            # Evaluate the character error rate
            # Compute the char error rate
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()

    def perform_training(self, config):

        print("Start transformer!")

        # get GPU if available otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # generate folder for weights folder if it does not exist yet
        Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

        # get data loaders and tokenizers
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = create_language_datasets(config)

        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

        # get tensorboard writer
        writer = SummaryWriter(config['experiment_name'])

        # specify Adom optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

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

        for epoch in range(initial_epoch, config['num_epochs']):
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

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            # Run validation at the end of every epoch
            self.__run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                                lambda msg: batch_iterator.write(msg), global_step, writer)

            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


