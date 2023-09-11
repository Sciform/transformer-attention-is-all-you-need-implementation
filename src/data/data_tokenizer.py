from pathlib import Path

# from HuggingFace
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_all_text_sequences_form_dataset_in_language(lang_data_set, language):
    for item in lang_data_set:
        yield item['translation'][language]


def get_or_build_tokenizer(config, lang_data_set, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        world_level_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_text_sequences_form_dataset_in_language(lang_data_set, language),
                                      trainer=world_level_trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
