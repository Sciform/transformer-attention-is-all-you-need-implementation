from sci_tf.trainer.transformer_trainer import TransformerTrainer
from sci_tf.config.project_config import Config
import warnings
import logging

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler("logger-file.log"),
                              logging.StreamHandler()])


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # get project config
    config = Config()

    config.MODEL['batch_size'] = 8
    config.MODEL['pretrained_model_epoch'] = None
    config.MODEL['num_epochs'] = 1

    logging.info('Main: build a transformer model and perform training')

    # create trainer and perform training
    transformer_trainer = TransformerTrainer(config)
    transformer_trainer.perform_training()
