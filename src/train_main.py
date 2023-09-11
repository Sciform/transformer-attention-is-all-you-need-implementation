import warnings
import logging
logging.basicConfig(level=logging.INFO)

from src.config.model_config import get_model_config
from src.trainer.transformer_trainer import TransformerTrainer

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # get model config
    model_config = get_model_config()

    logging.info('Main: build a transformer model and perform training')

    # create trainer and perform training
    transformer_trainer = TransformerTrainer()
    transformer_trainer.perform_training(model_config)
