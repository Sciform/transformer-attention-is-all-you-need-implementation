import logging

import torch


def get_proc_device():
    
    device = torch.device("cuda-gpu" if torch.cuda.is_available() else "cpu")
    logging.info(f'Tf_utils: The processing unit is {device}')
    
    return device