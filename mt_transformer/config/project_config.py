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
            "learning_rate": 10**-4,
            "d_model": 512,
            "pretrained_model_epoch": None # starting from None, 0, 1,
        }
        
        self.INFERENCE = {
            "trained_model_epoch": 29
        }
        
        self.__dictionary_folder = "data"
        self.__dictionary_filename = "dictionary_{0}.json"
        
        self.__experiment_folder = "experiments"
        self.__experiment_filename = "mt_transformer_exp"

        self.__saved_model_folder = "saved_model"
        self.__saved_model_filename = "mt_transformer_model_"
        
        
    def set_dictionary_folder_path(self, dictionary_folder_path: str) -> None:
        self.__dictionary_folder = dictionary_folder_path
        
        
    def set_saved_model_folder_path(self, saved_model_folder_path: str) -> None:
        self.__saved_model_folder = saved_model_folder_path
        
        
    def get_rel_dictionary_file_path(self, language: str) -> Path:
        
        Path(self.__dictionary_folder).mkdir(parents=True, exist_ok=True)
        full_dictionary_file_path = Path('.') / Path(self.__dictionary_folder) / Path(self.__dictionary_filename.format(language))
        print(f"full_dictionary_file_path = {full_dictionary_file_path}")
            
        return full_dictionary_file_path
    
    
    def get_experiments_file_path(self) -> Path:
        
        Path(self.__experiment_folder).mkdir(parents=True, exist_ok=True)
        full_exp_path = Path('.') / Path(self.__experiment_folder) / Path(self.__experiment_filename)
            
        return full_exp_path
        

    def get_saved_model_file_path(self, epoch: str) -> Path:
        
        Path(self.__saved_model_folder).mkdir(parents=True, exist_ok=True)
            
        model_filename = f"{self.__saved_model_filename}{epoch}.pt"
        full_saved_model_path = Path('.') / Path(self.__saved_model_folder) / Path(model_filename)  
        print(f'full_saved_model_path = {full_saved_model_path}')

        return full_saved_model_path
    