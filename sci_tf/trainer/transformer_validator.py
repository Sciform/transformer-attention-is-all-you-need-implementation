import os

import torch
import torchmetrics.text as tmtext

from sci_tf.model.greedy_decoder import GreedyDecoder


class TransformerValidator:

    def __init__(self) -> None:
        pass

    def perform_validation(self,
                           model,
                           validation_ds,
                           tokenizer_tgt,
                           max_len,
                           device,
                           print_msg,
                           global_step,
                           writer,
                           num_examples=2) -> None:

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
        except ImportError:
            # if we can't get the console width, use 80 as default
            console_width = 80

        with torch.no_grad():
            for batch in validation_ds:
                count += 1
                # (b, seq_len)
                encoder_input = batch["encoder_input"].to(device)
                # (b, 1, 1, seq_len)
                encoder_mask = batch["encoder_mask"].to(device)

                # check that the batch size is 1
                assert encoder_input.size(
                    0) == 1, "Batch size must be >= 1 for validation"

                greedy_decoder = GreedyDecoder()
                model_out = greedy_decoder.greedy_decode(
                    model, encoder_input, encoder_mask, tokenizer_tgt, max_len,
                    device)

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(
                    model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)

                # Print the source, target and model output
                print_msg('-' * console_width)
                print_msg(f"{f'Source: ':>12}{source_text}")
                print_msg(f"{f'Target: ':>12}{target_text}")
                print_msg(f"{f'Predicted: ':>12}{model_out_text}")

                if count == num_examples:
                    print_msg('-' * console_width)
                    break

        if writer is not None:
            # Evaluate character error rate
            cer_metric = tmtext.CharErrorRate()
            cer = cer_metric(predicted, expected)
            writer.add_scalar('validation CER', cer, global_step)
            writer.flush()

            # Evaluate word error rate
            wer_metric = tmtext.WordErrorRate()
            wer = wer_metric(predicted, expected)
            writer.add_scalar('validation WER', wer, global_step)
            writer.flush()

            # Evaluate the BLEU metric
            bleu_metric = tmtext.BLEUScore()
            bleu = bleu_metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()
