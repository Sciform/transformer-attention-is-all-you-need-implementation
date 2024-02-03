import logging

# huggingface datasets
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
# pytorch
from torch.utils.data import DataLoader, random_split

from sci_tf.config.project_config import Config
from sci_tf.data_handler.data_tokenizer import get_or_create_tokenizer
from sci_tf.data_handler.two_language_data_set import TwoLanguagesDataset


def get_raw_data_opus_books(config: Config) -> Dataset:
    """Load raw data set \"Opus Books\"

    :param config: Configuration
    :type config: Config
    :return: Raw data set "Opus Books" from Hugging Face
    :rtype: datasets.Dataset
    """

    # for the opus books only the train split is available
    data_set = load_dataset(
        'opus_books', f"{config.DATA['lang_src']}-{config.DATA['lang_tgt']}",
        split='train')

    logging.info("Opus books raw data set loaded for source language %s and " +
                 "target language %s", config.DATA['lang_src'],
                 config.DATA['lang_tgt'])

    return data_set


def create_tokenizers_dataloaders(config: Config) -> [DataLoader, DataLoader,
                                                      Tokenizer, Tokenizer]:
    """Create tokenizers and data loaders

    :param config: Configuration
    :type config: Config
    :return: Data loader for training and validation data, Tokenizers for 
            source and target vocabulary 
    :rtype: [DataLoader, DataLoader, Tokenizer, Tokenizer]
    """
    # load opus_books data set (it has only a train split)
    ds_raw = get_raw_data_opus_books(config)

    # create tokenizers
    tokenizer_src = get_or_create_tokenizer(
        config, ds_raw, config.DATA['lang_src'])
    tokenizer_tgt = get_or_create_tokenizer(
        config, ds_raw, config.DATA['lang_tgt'])

    logging.info('data_loader: tokenizers created')

    # split into 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size])

    # get train and val data set
    train_ds = TwoLanguagesDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                   config.DATA['lang_src'],
                                   config.DATA['lang_tgt'],
                                   config.DATA['seq_len'])
    val_ds = TwoLanguagesDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                                 config.DATA['lang_src'],
                                 config.DATA['lang_tgt'],
                                 config.DATA['seq_len'])

    logging.info('data_loader: train and validation data set created')

    # create PyTorch train and val dataloaders
    train_dataloader = DataLoader(
        train_ds, batch_size=config.MODEL['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    logging.info('data_loader: dataloader and tokenizers created')

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
