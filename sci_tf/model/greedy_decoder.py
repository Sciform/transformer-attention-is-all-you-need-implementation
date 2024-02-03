import torch

from sci_tf.data_handler.masks import causal_mask


class GreedyDecoder:

    def __init__(self) -> None:
        pass

    def greedy_decode(self,
                      transformer_model,
                      source,
                      source_mask,
                      tokenizer_tgt,
                      max_len,
                      device) -> torch.Tensor:

        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = transformer_model.encode(source, source_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(
            sos_idx).type_as(source).to(device)

        while True:
            if decoder_input.size(1) == max_len:
                break

            # build mask for target
            decoder_mask = causal_mask(
                decoder_input.size(1)).type_as(source_mask).to(device)

            # calculate output
            decoded_output = transformer_model.decode(
                encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = transformer_model.project(decoded_output[:, -1])

            _, next_word = torch.max(prob, dim=1)

            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(
                    source).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)
