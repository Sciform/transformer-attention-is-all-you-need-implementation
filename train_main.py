import warnings
import logging
logging.basicConfig(level=logging.INFO)

from mt_transformer.config.project_config import Config
from mt_transformer.trainer.transformer_trainer import TransformerTrainer

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # get model config
    config = Config()

    logging.info('Main: build a transformer model and perform training')

    # create trainer and perform training
    transformer_trainer = TransformerTrainer()
    transformer_trainer.perform_training(config)
