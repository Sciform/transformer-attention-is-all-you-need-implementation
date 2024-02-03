import logging
from typing import Any, Generator

# from HuggingFace
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from sci_tf.config.project_config import Config


def get_all_text_sequences_from_dataset_in_language(
        ds_raw: Dataset,
        language: str) -> Generator[Any, Any, None]:
    """Get all text sequences form a data set in a given language

    :param ds_raw: _description_
    :type ds_raw: Dataset
    :param language: language
    :type language: str
    :yield: iterator over every sentence in data set
    :rtype: Generator[Any, Any, None]
    """
    for item in ds_raw:
        yield item['translation'][language]


def check_max_seq_length(ds_raw: Dataset,
                         tokenizer: Tokenizer,
                         language: str) -> None:
    """Check for maximum sequence length

    :param ds_raw: raw data set "Opus Books"
    :type ds_raw: Dataset
    :param tokenizer: tokenizer
    :type tokenizer: Tokenizer
    :param language: language
    :type language: str
    """
    # determine the maximum length of each sentence in the source and target language
    max_len = 0

    for item in ds_raw:
        ids = tokenizer.encode(item['translation'][language]).ids
        max_len = max(max_len, len(ids))

    logging.info("Max length of a sentence in language %s is %s.", language,
                 max_len)


def get_or_create_tokenizer(config: Config,
                            ds_raw: Dataset,
                            language: str) -> Tokenizer:
    """Get or create Hugging Face tokenizer 

    :param config: configuration
    :type config: Config
    :param ds_raw: raw data set "Opus Books"
    :type ds_raw: Dataset
    :param language: language
    :type language: str
    :return: tokenizer for provided data set in given language
    :rtype: Tokenizer
    """

    tokenizer_path = config.get_rel_dictionary_file_path(language)

    if not tokenizer_path.exists():

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        world_level_trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_text_sequences_from_dataset_in_language(ds_raw, language),
            trainer=world_level_trainer)

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    check_max_seq_length(ds_raw, tokenizer, language)

    return tokenizer
