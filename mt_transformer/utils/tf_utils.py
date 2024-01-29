"""Imports"""
import logging
import torch


def get_proc_device() -> str:
    """
    get processor type either "cuda" or "cpu"

    Returns:
        string: device is either "cuda" or "cpu"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Tf_utils: The processing unit is %s, device")
    return device
