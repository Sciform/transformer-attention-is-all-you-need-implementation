from pathlib import Path


class Config():
    
    def __init__(self) -> None:
        
        self.DATA = {
            "lang_src": "en",
            "lang_tgt": "it",
            "seq_len": 480,
        }
        
        self.MODEL = {
            "batch_size": 8,        # adjust if too large for GPU
            "num_epochs": 3,
            "lr": 10**-4,
            "d_model": 512,
            "preload": None
        }
        
        self.__dictionary_folder = "data"
        self.__dictionary_filename = "dictionary_{0}.json"
        self.__experiment_folder = "experiments"
        self.__experiment_filename = "mt_transformer_exp"
        self.__saved_model_folder = "saved_model"
        self.__saved_model_filename = "mt_transformer_model_"
        
        
    def get_rel_dictionary_file_path(self, language: str) -> Path:
        
        Path(self.__dictionary_folder).mkdir(parents=True, exist_ok=True)
            
        return Path('.') / Path(self.__dictionary_folder) / Path(self.__dictionary_filename.format(language))
    
    
    def get_experiments_file_path(self) -> Path:
        
        Path(self.__experiment_folder).mkdir(parents=True, exist_ok=True)
            
        return Path('.') / Path(self.__experiment_folder) / Path(self.__experiment_filename)
        

    def get_saved_model_file_path(self, epoch: str) -> Path:
        
        Path(self.__saved_model_folder).mkdir(parents=True, exist_ok=True)
            
        model_filename = f"{self.__saved_model_filename}{epoch}.pt"
        return Path('.') / Path(self.__saved_model_folder) / Path(model_filename)  
    
