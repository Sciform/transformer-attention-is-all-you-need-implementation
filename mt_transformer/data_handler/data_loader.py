
import logging
# huggingface datasets
from datasets import load_dataset
# pytorch
from torch.utils.data import DataLoader, random_split

from mt_transformer.data_handler.data_tokenizer import get_or_create_tokenizer
from mt_transformer.data_handler.two_language_data_set import TwoLanguagesDataset


def get_raw_data_opus_books(config):

    # for the opus books only the train split is available
    return load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    logging.info(f'Opus books raw data set loaded.')


def create_tokenizers_dataloaders(config):

    # load opus_books data set (it has only a train split)
    ds_raw = get_raw_data_opus_books(config)

    # create tokenizers
    tokenizer_src = get_or_create_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_create_tokenizer(config, ds_raw, config['lang_tgt'])

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

