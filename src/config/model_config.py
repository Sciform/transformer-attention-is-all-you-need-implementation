from pathlib import Path


def get_model_config():
    return {
        "lang_src": "de",
        "lang_tgt": "en",
        "batch_size": 2,        # adjust if too large for GPU
        "num_epochs": 3,
        "lr": 10**-4,
        "seq_len": 480,
        "d_model": 512,
        "dictionary_file": "dictionary_{0}.json",
        "experiment_name": "experiments/transf_model",
        "saved_model_folder": "../saved_model",
        "model_basename": "transf_model_",
        "preload": None
    }


def get_saved_model_file_path(config, epoch: str):
    saved_model_folder = config["saved_model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / saved_model_folder / model_filename)
