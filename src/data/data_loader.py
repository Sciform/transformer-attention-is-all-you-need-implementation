
import logging
# huggingface datasets
from datasets import load_dataset
# pytorch
from torch.utils.data import DataLoader, random_split

from src.data.data_tokenizer import get_or_create_tokenizer
from src.data.two_language_data_set import TwoLanguagesDataset


def create_tokenizers_dataloaders(config):

    # load opus_books data set (it has only a train split)
    ds_raw = get_raw_data_opus_books(config)

    # create tokenizers
    tokenizer_src = get_or_create_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_create_tokenizer(config, ds_raw, config['lang_tgt'])

    # determine the maximum length of each sentence in the source and target language
    """
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    logging.info(f'Max length of source language sentence: {max_len_src}')
    logging.info(f'Max length of target language sentence: {max_len_tgt}')
    """

    # split into 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # get train and val data set
    train_ds = TwoLanguagesDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                   config['seq_len'])
    val_ds = TwoLanguagesDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                 config['seq_len'])

    # create PyTorch train and val dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_raw_data_opus_books(config):

    # for the opus books only the train split is available
    return load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    logging.info(f'Opus books raw data set loaded.')

