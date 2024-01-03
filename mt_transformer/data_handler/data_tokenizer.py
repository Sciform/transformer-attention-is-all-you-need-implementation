import logging
from pathlib import Path

# from HuggingFace
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


def get_all_text_sequences_form_dataset_in_language(lang_data_set, language):
    for item in lang_data_set:
        yield item['translation'][language]


def check_max_seq_length(ds_raw, tokenizer, language):
    # determine the maximum length of each sentence in the source and target language
    max_len = 0

    for item in ds_raw:
        ids = tokenizer.encode(item['translation'][language]).ids
        max_len = max(max_len, len(ids))

    logging.info(f'Max length of a sentence in language {language} is: {max_len}')


def get_or_create_tokenizer(config, ds_raw, language):
    tokenizer_path = Path("data \ "+ config['dictionary_file'].format(language))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        world_level_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_text_sequences_form_dataset_in_language(ds_raw, language),
                                      trainer=world_level_trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    check_max_seq_length(ds_raw, tokenizer, language)

    return tokenizer

