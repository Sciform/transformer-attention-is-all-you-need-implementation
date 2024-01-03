import warnings
import logging

logging.basicConfig(level=logging.INFO, 
                    handlers=[logging.FileHandler("logger-file.log"),
                              logging.StreamHandler()])


from mt_transformer.config.project_config import Config
from mt_transformer.trainer.transformer_trainer import TransformerTrainer

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")

    # get project config
    config = Config()
    
    config.MODEL['batch_size'] = 6
    config.MODEL['preload'] = None
    config.MODEL['num_epochs'] = 30
    
    logging.info('Main: build a transformer model and perform training')

    # create trainer and perform training
    transformer_trainer = TransformerTrainer()
    transformer_trainer.perform_training(config)
