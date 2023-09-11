__author__ = '{Ursula Maria Mayer}'
__copyright__ = 'Copyright {2023}, {Sciform GmbH}'

import os

import torch
import torchmetrics


class TransformerValidator:

    def perform_validation(self, model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg,
                           global_step, writer, num_examples=2):

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
            # if we can't get the console width, use 80 as default
            console_width = 80

        with torch.no_grad():
            for batch in validation_ds:
                count += 1
                encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
                encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

                # check that the batch size is 1
                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out = self.__greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt,
                                                 max_len, device)

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

        if writer is not None:
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
            decoder_mask = self.__causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

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

    def __causal_mask(self, size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

