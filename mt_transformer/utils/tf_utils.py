import logging
import torch


def get_proc_device() -> str:
    """Get processor type either "cuda" or "cpu"

    :return: device string is either "cuda" or "cpu"
    :rtype: str
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Tf_utils: The processing unit is %s", device)
    return device
