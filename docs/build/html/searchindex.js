Search.setIndex({"alltitles":{"How to implement a Transformer ?":[[47,null]],"Introduction":[[48,null]],"Layer Types":[[50,null]],"My take on scaling the embeddings":[[49,"my-take-on-scaling-the-embeddings"]],"Quick Select":[[47,"quick-select"]],"Sciform - AI Consulting":[[47,"sciform-ai-consulting"]],"The Embeddings Layer":[[49,null]],"Transformer":[[50,null]],"What are embeddings ?":[[49,"what-are-embeddings"]],"sci_tf":[[0,null]],"sci_tf.config":[[1,null]],"sci_tf.config.project_config":[[2,null]],"sci_tf.config.project_config.Config":[[3,null]],"sci_tf.data_handler":[[4,null]],"sci_tf.data_handler.data_loader":[[5,null]],"sci_tf.data_handler.data_loader.create_tokenizers_dataloaders":[[6,null]],"sci_tf.data_handler.data_loader.get_raw_data_opus_books":[[7,null]],"sci_tf.data_handler.data_tokenizer":[[8,null]],"sci_tf.data_handler.data_tokenizer.check_max_seq_length":[[9,null]],"sci_tf.data_handler.data_tokenizer.get_all_text_sequences_from_dataset_in_language":[[10,null]],"sci_tf.data_handler.data_tokenizer.get_or_create_tokenizer":[[11,null]],"sci_tf.data_handler.masks":[[12,null]],"sci_tf.data_handler.masks.causal_mask":[[13,null]],"sci_tf.data_handler.two_language_data_set":[[14,null]],"sci_tf.data_handler.two_language_data_set.TwoLanguagesDataset":[[15,null]],"sci_tf.inference":[[16,null]],"sci_tf.inference.tf_inference":[[17,null]],"sci_tf.inference.tf_inference.TfInference":[[18,null]],"sci_tf.inference.tf_visualizer":[[19,null]],"sci_tf.inference.tf_visualizer.TfVisualizer":[[20,null]],"sci_tf.model":[[21,null]],"sci_tf.model.greedy_decoder":[[22,null]],"sci_tf.model.greedy_decoder.GreedyDecoder":[[23,null]],"sci_tf.model.layers":[[24,null]],"sci_tf.model.layers.FeedForwardBlock":[[25,null]],"sci_tf.model.layers.LayerNormalization":[[26,null]],"sci_tf.model.layers.MultiHeadAttention":[[27,null]],"sci_tf.model.layers.PositionalEncoding":[[28,null]],"sci_tf.model.layers.ProjectionLayer":[[29,null]],"sci_tf.model.layers.ResidualConnection":[[30,null]],"sci_tf.model.layers.TokenEmbeddings":[[31,null]],"sci_tf.model.transformer_model":[[32,null]],"sci_tf.model.transformer_model.Decoder":[[33,null]],"sci_tf.model.transformer_model.DecoderStack":[[34,null]],"sci_tf.model.transformer_model.Encoder":[[35,null]],"sci_tf.model.transformer_model.EncoderStack":[[36,null]],"sci_tf.model.transformer_model.TransformerModel":[[37,null]],"sci_tf.trainer":[[38,null]],"sci_tf.trainer.transformer_trainer":[[39,null]],"sci_tf.trainer.transformer_trainer.TransformerTrainer":[[40,null]],"sci_tf.trainer.transformer_validator":[[41,null]],"sci_tf.trainer.transformer_validator.TransformerValidator":[[42,null]],"sci_tf.utils":[[43,null]],"sci_tf.utils.tf_utils":[[44,null]],"sci_tf.utils.tf_utils.get_proc_device":[[45,null]]},"docnames":["_autosummary/sci_tf","_autosummary/sci_tf.config","_autosummary/sci_tf.config.project_config","_autosummary/sci_tf.config.project_config.Config","_autosummary/sci_tf.data_handler","_autosummary/sci_tf.data_handler.data_loader","_autosummary/sci_tf.data_handler.data_loader.create_tokenizers_dataloaders","_autosummary/sci_tf.data_handler.data_loader.get_raw_data_opus_books","_autosummary/sci_tf.data_handler.data_tokenizer","_autosummary/sci_tf.data_handler.data_tokenizer.check_max_seq_length","_autosummary/sci_tf.data_handler.data_tokenizer.get_all_text_sequences_from_dataset_in_language","_autosummary/sci_tf.data_handler.data_tokenizer.get_or_create_tokenizer","_autosummary/sci_tf.data_handler.masks","_autosummary/sci_tf.data_handler.masks.causal_mask","_autosummary/sci_tf.data_handler.two_language_data_set","_autosummary/sci_tf.data_handler.two_language_data_set.TwoLanguagesDataset","_autosummary/sci_tf.inference","_autosummary/sci_tf.inference.tf_inference","_autosummary/sci_tf.inference.tf_inference.TfInference","_autosummary/sci_tf.inference.tf_visualizer","_autosummary/sci_tf.inference.tf_visualizer.TfVisualizer","_autosummary/sci_tf.model","_autosummary/sci_tf.model.greedy_decoder","_autosummary/sci_tf.model.greedy_decoder.GreedyDecoder","_autosummary/sci_tf.model.layers","_autosummary/sci_tf.model.layers.FeedForwardBlock","_autosummary/sci_tf.model.layers.LayerNormalization","_autosummary/sci_tf.model.layers.MultiHeadAttention","_autosummary/sci_tf.model.layers.PositionalEncoding","_autosummary/sci_tf.model.layers.ProjectionLayer","_autosummary/sci_tf.model.layers.ResidualConnection","_autosummary/sci_tf.model.layers.TokenEmbeddings","_autosummary/sci_tf.model.transformer_model","_autosummary/sci_tf.model.transformer_model.Decoder","_autosummary/sci_tf.model.transformer_model.DecoderStack","_autosummary/sci_tf.model.transformer_model.Encoder","_autosummary/sci_tf.model.transformer_model.EncoderStack","_autosummary/sci_tf.model.transformer_model.TransformerModel","_autosummary/sci_tf.trainer","_autosummary/sci_tf.trainer.transformer_trainer","_autosummary/sci_tf.trainer.transformer_trainer.TransformerTrainer","_autosummary/sci_tf.trainer.transformer_validator","_autosummary/sci_tf.trainer.transformer_validator.TransformerValidator","_autosummary/sci_tf.utils","_autosummary/sci_tf.utils.tf_utils","_autosummary/sci_tf.utils.tf_utils.get_proc_device","api","index","intro","transformer/embeddings","transformer/index"],"envversion":{"nbsphinx":4,"sphinx":65,"sphinx.domains.c":3,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":9,"sphinx.domains.index":1,"sphinx.domains.javascript":3,"sphinx.domains.math":2,"sphinx.domains.python":4,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1},"filenames":["_autosummary\\sci_tf.rst","_autosummary\\sci_tf.config.rst","_autosummary\\sci_tf.config.project_config.rst","_autosummary\\sci_tf.config.project_config.Config.rst","_autosummary\\sci_tf.data_handler.rst","_autosummary\\sci_tf.data_handler.data_loader.rst","_autosummary\\sci_tf.data_handler.data_loader.create_tokenizers_dataloaders.rst","_autosummary\\sci_tf.data_handler.data_loader.get_raw_data_opus_books.rst","_autosummary\\sci_tf.data_handler.data_tokenizer.rst","_autosummary\\sci_tf.data_handler.data_tokenizer.check_max_seq_length.rst","_autosummary\\sci_tf.data_handler.data_tokenizer.get_all_text_sequences_from_dataset_in_language.rst","_autosummary\\sci_tf.data_handler.data_tokenizer.get_or_create_tokenizer.rst","_autosummary\\sci_tf.data_handler.masks.rst","_autosummary\\sci_tf.data_handler.masks.causal_mask.rst","_autosummary\\sci_tf.data_handler.two_language_data_set.rst","_autosummary\\sci_tf.data_handler.two_language_data_set.TwoLanguagesDataset.rst","_autosummary\\sci_tf.inference.rst","_autosummary\\sci_tf.inference.tf_inference.rst","_autosummary\\sci_tf.inference.tf_inference.TfInference.rst","_autosummary\\sci_tf.inference.tf_visualizer.rst","_autosummary\\sci_tf.inference.tf_visualizer.TfVisualizer.rst","_autosummary\\sci_tf.model.rst","_autosummary\\sci_tf.model.greedy_decoder.rst","_autosummary\\sci_tf.model.greedy_decoder.GreedyDecoder.rst","_autosummary\\sci_tf.model.layers.rst","_autosummary\\sci_tf.model.layers.FeedForwardBlock.rst","_autosummary\\sci_tf.model.layers.LayerNormalization.rst","_autosummary\\sci_tf.model.layers.MultiHeadAttention.rst","_autosummary\\sci_tf.model.layers.PositionalEncoding.rst","_autosummary\\sci_tf.model.layers.ProjectionLayer.rst","_autosummary\\sci_tf.model.layers.ResidualConnection.rst","_autosummary\\sci_tf.model.layers.TokenEmbeddings.rst","_autosummary\\sci_tf.model.transformer_model.rst","_autosummary\\sci_tf.model.transformer_model.Decoder.rst","_autosummary\\sci_tf.model.transformer_model.DecoderStack.rst","_autosummary\\sci_tf.model.transformer_model.Encoder.rst","_autosummary\\sci_tf.model.transformer_model.EncoderStack.rst","_autosummary\\sci_tf.model.transformer_model.TransformerModel.rst","_autosummary\\sci_tf.trainer.rst","_autosummary\\sci_tf.trainer.transformer_trainer.rst","_autosummary\\sci_tf.trainer.transformer_trainer.TransformerTrainer.rst","_autosummary\\sci_tf.trainer.transformer_validator.rst","_autosummary\\sci_tf.trainer.transformer_validator.TransformerValidator.rst","_autosummary\\sci_tf.utils.rst","_autosummary\\sci_tf.utils.tf_utils.rst","_autosummary\\sci_tf.utils.tf_utils.get_proc_device.rst","api.rst","index.md","intro.rst","transformer\\embeddings.rst","transformer\\index.md"],"indexentries":{"feedforwardblock (class in sci_tf.model.layers)":[[25,"sci_tf.model.layers.FeedForwardBlock",false]],"forward() (positionalencoding method)":[[28,"sci_tf.model.layers.PositionalEncoding.forward",false]],"forward() (tokenembeddings method)":[[31,"sci_tf.model.layers.TokenEmbeddings.forward",false]],"layernormalization (class in sci_tf.model.layers)":[[26,"sci_tf.model.layers.LayerNormalization",false]],"module":[[21,"module-sci_tf.model",false],[24,"module-sci_tf.model.layers",false]],"multiheadattention (class in sci_tf.model.layers)":[[27,"sci_tf.model.layers.MultiHeadAttention",false]],"positionalencoding (class in sci_tf.model.layers)":[[28,"sci_tf.model.layers.PositionalEncoding",false]],"projectionlayer (class in sci_tf.model.layers)":[[29,"sci_tf.model.layers.ProjectionLayer",false]],"residualconnection (class in sci_tf.model.layers)":[[30,"sci_tf.model.layers.ResidualConnection",false]],"sci_tf.model":[[21,"module-sci_tf.model",false]],"sci_tf.model.layers":[[24,"module-sci_tf.model.layers",false]],"tokenembeddings (class in sci_tf.model.layers)":[[31,"sci_tf.model.layers.TokenEmbeddings",false]]},"objects":{"":[[0,0,0,"-","sci_tf"]],"sci_tf":[[1,0,0,"-","config"],[4,0,0,"-","data_handler"],[16,0,0,"-","inference"],[21,0,0,"-","model"],[38,0,0,"-","trainer"],[43,0,0,"-","utils"]],"sci_tf.config":[[2,0,0,"-","project_config"]],"sci_tf.config.project_config":[[3,1,1,"","Config"]],"sci_tf.config.project_config.Config":[[3,2,1,"","get_experiments_file_path"],[3,2,1,"","get_rel_dictionary_file_path"],[3,2,1,"","get_saved_model_file_path"]],"sci_tf.data_handler":[[5,0,0,"-","data_loader"],[8,0,0,"-","data_tokenizer"],[12,0,0,"-","masks"],[14,0,0,"-","two_language_data_set"]],"sci_tf.data_handler.data_loader":[[6,3,1,"","create_tokenizers_dataloaders"],[7,3,1,"","get_raw_data_opus_books"]],"sci_tf.data_handler.data_tokenizer":[[9,3,1,"","check_max_seq_length"],[10,3,1,"","get_all_text_sequences_from_dataset_in_language"],[11,3,1,"","get_or_create_tokenizer"]],"sci_tf.data_handler.masks":[[13,3,1,"","causal_mask"]],"sci_tf.data_handler.two_language_data_set":[[15,1,1,"","TwoLanguagesDataset"]],"sci_tf.inference":[[17,0,0,"-","tf_inference"],[19,0,0,"-","tf_visualizer"]],"sci_tf.inference.tf_inference":[[18,1,1,"","TfInference"]],"sci_tf.inference.tf_visualizer":[[20,1,1,"","TfVisualizer"]],"sci_tf.model":[[22,0,0,"-","greedy_decoder"],[24,0,0,"-","layers"],[32,0,0,"-","transformer_model"]],"sci_tf.model.greedy_decoder":[[23,1,1,"","GreedyDecoder"]],"sci_tf.model.layers":[[25,1,1,"","FeedForwardBlock"],[26,1,1,"","LayerNormalization"],[27,1,1,"","MultiHeadAttention"],[28,1,1,"","PositionalEncoding"],[29,1,1,"","ProjectionLayer"],[30,1,1,"","ResidualConnection"],[31,1,1,"","TokenEmbeddings"]],"sci_tf.model.layers.PositionalEncoding":[[28,2,1,"","forward"]],"sci_tf.model.layers.TokenEmbeddings":[[31,2,1,"","forward"]],"sci_tf.model.transformer_model":[[33,1,1,"","Decoder"],[34,1,1,"","DecoderStack"],[35,1,1,"","Encoder"],[36,1,1,"","EncoderStack"],[37,1,1,"","TransformerModel"]],"sci_tf.model.transformer_model.DecoderStack":[[34,2,1,"","forward"]],"sci_tf.model.transformer_model.EncoderStack":[[36,2,1,"","forward"]],"sci_tf.trainer":[[39,0,0,"-","transformer_trainer"],[41,0,0,"-","transformer_validator"]],"sci_tf.trainer.transformer_trainer":[[40,1,1,"","TransformerTrainer"]],"sci_tf.trainer.transformer_trainer.TransformerTrainer":[[40,2,1,"","perform_training"]],"sci_tf.trainer.transformer_validator":[[42,1,1,"","TransformerValidator"]],"sci_tf.utils":[[44,0,0,"-","tf_utils"]],"sci_tf.utils.tf_utils":[[45,3,1,"","get_proc_device"]]},"objnames":{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},"objtypes":{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},"terms":{"0":[31,37,49],"03762":[0,47,48],"06":26,"1":[31,37,49],"1706":[0,47,48],"1e":26,"2017":[0,47,48,50],"2048":37,"6":37,"8":37,"87906":[],"A":[0,28,31,47,48,50],"For":31,"In":[31,49],"It":[31,49],"Or":[],"The":[28,31,40,47,48,50],"_autosummari":47,"_description_":10,"ab":[0,47,48],"about":47,"actual":36,"ad":[],"add":[28,30],"al":[0,47,48,50],"all":[0,10,47,48,50],"almost":[47,48],"alreadi":[31,49],"an":[47,48],"ani":[10,28,31],"appli":[28,31,49],"ar":[28,31,50],"architectur":[0,47,48,50],"art":[47,48],"artifici":47,"arxiv":[0,47,48],"attent":[0,27,47,48,50],"attribut":[25,26,27,28,29,30,31,33,34,35,36,37],"avoid":36,"base":[3,15,18,20,23,25,26,27,28,29,30,31,33,34,35,36,37,40,42],"batch":[26,28,31],"been":[31,49],"befor":49,"being":49,"between":[],"book":[7,9,11],"bring":[31,49],"build":50,"can":[31,47,49],"captur":49,"causal":34,"charact":49,"check":[9,30,47],"class":[2,3,14,15,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39,40,41,42],"closer":[31,49],"code":47,"com":47,"comput":[31,47],"config":[6,7,11,18,20,37,40],"configur":[3,6,7,11,40],"consider":[31,49],"construct":3,"contact":47,"contain":28,"cpu":45,"creat":[6,11],"cuda":45,"current":[30,47,48],"d_":[31,49],"d_ff":[25,33,34,35,36,37],"d_model":[25,27,28,29,31,33,34,35,36],"data":[6,7,9,10,11,31,49],"dataload":6,"datasci":[],"dataset":[7,9,10,11,15],"deep":[47,48],"dens":49,"detail":47,"devic":45,"dictionari":[3,31],"dictionary_s":31,"dim":[28,31],"dimens":31,"discuss":[31,49],"distribut":[31,49],"do":49,"document":[0,47],"doe":[],"drop":[27,28],"drop_out_prob":28,"dropout":[25,30],"dropout_prob":[27,28,33,34,35,36,37],"ds_raw":[9,10,11,15],"e":[31,49],"each":49,"either":45,"embed":[27,31,50],"encod":[28,31,49],"encoder_output":34,"ep":26,"epoch":3,"establish":34,"et":[0,47,48,50],"everi":[10,28,31,47,48],"exchang":[31,49],"experi":3,"explain":[31,49],"explicitli":[31,49],"ext":31,"f":[],"face":[7,11],"factor":[31,49],"fals":28,"featur":[27,28,31,49],"file":3,"find":47,"first":30,"fix":31,"float":[27,28],"flow":40,"form":10,"forward":[28,31,34,36],"found":[31,49],"frac":49,"fract":[31,49],"from":[7,31,48,49],"function":[5,8,12,44],"gaussian":[31,49],"gener":10,"get":[3,10,11,45,47],"get_experiments_file_path":3,"get_rel_dictionary_file_path":3,"get_saved_model_file_path":3,"given":[10,11],"gmbh":47,"googl":49,"h":[33,34,35,36,37],"handl":40,"have":[31,49],"head":27,"here":[47,48],"hold":[28,31],"how":[48,50],"howev":49,"http":[0,26,47,48],"hug":[7,11],"i":[0,28,31,45,47,48,49,50],"implement":[0,48,49],"index":47,"initi":[31,49],"inlin":[],"input":[28,49],"int":[27,28,31],"intellig":47,"interact":36,"io":26,"iter":10,"k":[],"languag":[3,9,10,11],"learn":[26,28,31,47,48,50],"length":[9,28,31],"like":47,"load":[3,7],"loader":6,"machin":50,"manag":47,"map":[31,49],"mask":36,"math":31,"mathcal":[31,49],"matrix":[31,49],"maximum":9,"md":47,"mean":[31,49],"mention":49,"method":[3,15,18,20,23,25,26,27,28,29,30,31,33,34,35,36,37,40,42],"model":[3,40,47,48,49],"modul":[25,26,27,28,29,30,31,33,34,35,36,37],"more":47,"mt_transform":40,"multi":27,"multipli":[31,49],"my":50,"n":[31,49],"necessari":[31,49],"need":[0,47,48,50],"neither":49,"none":[9,10],"nor":49,"norm":30,"normal":[26,30,31,49],"num_batch":31,"num_head":27,"num_stack":[33,35,37],"number":[27,28,31,49],"numer":49,"object":[3,18,20,23,40,42],"opu":[7,9,11],"org":[0,47,48],"origin":[0,31,47,48,49,50],"other":[31,47,49],"our":47,"out":[27,28,47],"over":10,"own":47,"pad":36,"paper":[0,31,49,50],"paramet":[3,6,7,9,10,11,27,28,31,34,36,40],"path":3,"per":[28,31],"perform":[31,40],"perform_train":40,"pinecon":26,"posit":[28,31,49],"power":[47,48],"precomput":28,"previou":[28,30],"probabl":[27,28],"process":49,"processor":45,"propos":[31,49,50],"provid":[11,47,48],"publish":[0,47,48],"purpos":[31,49],"pytorch":[31,49],"quantum":47,"question":[],"rac":31,"rang":[31,49],"raw":[7,9,11,49],"reason":[31,49],"refer":[47,49],"regard":[31,49],"rel":3,"relat":[31,49],"represent":49,"requires_grad_":28,"result":28,"return":[3,6,7,9,10,11,28,34,36,45],"rst":47,"rtype":31,"save":3,"scale":[31,50],"sci_tf":47,"scienc":[31,49],"scratch":48,"seem":[31,49],"semant":49,"sentenc":10,"seq_len":[15,28],"sequenc":[9,10,28,31],"sequence_length":31,"servic":47,"set":[7,9,10,11],"shape":28,"should":[],"similar":[31,49],"sinc":49,"size":[13,31],"so":[31,49],"solut":[47,50],"some":[31,49],"someth":[],"sourc":[3,6,7,9,10,11,13,15,18,20,23,25,26,27,28,29,30,31,33,34,35,36,37,40,42,45,49],"specif":3,"specifi":3,"sqrt":[31,49],"src_lang":15,"src_mask":[34,36],"src_vocab_s":37,"stack":[31,49],"stackexchang":[],"state":[47,48],"str":[3,9,10,11,45],"strateg":47,"string":45,"sublay":30,"subword":49,"sum_":[],"support":47,"t":[],"take":50,"target":6,"technic":47,"tell":28,"tensor":[28,31,49],"text":[10,49],"tgt_lang":15,"tgt_mask":34,"tgt_vocab_s":37,"therefor":[31,49],"thi":[31,49],"todo":[],"token":[6,9,11,28,31,36,49],"tokenizer_src":15,"tokenizer_tgt":15,"topic":[31,49],"train":[6,40],"transform":[0,48,49],"translat":50,"type":[3,6,7,9,10,11,28,45],"u":47,"understand":47,"us":[31,47,49],"usual":28,"valid":6,"valu":28,"varianc":[31,49],"variou":[31,49],"vaswani":[0,47,48,49,50],"vector":49,"versa":30,"vice":30,"vocab_s":29,"vocabulari":6,"wa":[31,49],"want":[],"we":[47,48],"well":49,"what":50,"where":[31,49],"which":28,"why":[],"word":49,"would":47,"write":[],"www":[26,47],"x":[28,31,34,36],"yield":10,"you":[0,47,48,50],"your":47},"titles":["sci_tf","sci_tf.config","sci_tf.config.project_config","sci_tf.config.project_config.Config","sci_tf.data_handler","sci_tf.data_handler.data_loader","sci_tf.data_handler.data_loader.create_tokenizers_dataloaders","sci_tf.data_handler.data_loader.get_raw_data_opus_books","sci_tf.data_handler.data_tokenizer","sci_tf.data_handler.data_tokenizer.check_max_seq_length","sci_tf.data_handler.data_tokenizer.get_all_text_sequences_from_dataset_in_language","sci_tf.data_handler.data_tokenizer.get_or_create_tokenizer","sci_tf.data_handler.masks","sci_tf.data_handler.masks.causal_mask","sci_tf.data_handler.two_language_data_set","sci_tf.data_handler.two_language_data_set.TwoLanguagesDataset","sci_tf.inference","sci_tf.inference.tf_inference","sci_tf.inference.tf_inference.TfInference","sci_tf.inference.tf_visualizer","sci_tf.inference.tf_visualizer.TfVisualizer","sci_tf.model","sci_tf.model.greedy_decoder","sci_tf.model.greedy_decoder.GreedyDecoder","sci_tf.model.layers","sci_tf.model.layers.FeedForwardBlock","sci_tf.model.layers.LayerNormalization","sci_tf.model.layers.MultiHeadAttention","sci_tf.model.layers.PositionalEncoding","sci_tf.model.layers.ProjectionLayer","sci_tf.model.layers.ResidualConnection","sci_tf.model.layers.TokenEmbeddings","sci_tf.model.transformer_model","sci_tf.model.transformer_model.Decoder","sci_tf.model.transformer_model.DecoderStack","sci_tf.model.transformer_model.Encoder","sci_tf.model.transformer_model.EncoderStack","sci_tf.model.transformer_model.TransformerModel","sci_tf.trainer","sci_tf.trainer.transformer_trainer","sci_tf.trainer.transformer_trainer.TransformerTrainer","sci_tf.trainer.transformer_validator","sci_tf.trainer.transformer_validator.TransformerValidator","sci_tf.utils","sci_tf.utils.tf_utils","sci_tf.utils.tf_utils.get_proc_device","&lt;no title&gt;","How to implement a Transformer ?","Introduction","The Embeddings Layer","Transformer"],"titleterms":{"The":49,"ai":47,"ar":49,"causal_mask":13,"check_max_seq_length":9,"config":[1,2,3],"consult":47,"create_tokenizers_dataload":6,"data_handl":[4,5,6,7,8,9,10,11,12,13,14,15],"data_load":[5,6,7],"data_token":[8,9,10,11],"decod":33,"decoderstack":34,"embed":49,"encod":35,"encoderstack":36,"feedforwardblock":25,"get_all_text_sequences_from_dataset_in_languag":10,"get_or_create_token":11,"get_proc_devic":45,"get_raw_data_opus_book":7,"greedy_decod":[22,23],"greedydecod":23,"how":47,"implement":47,"infer":[16,17,18,19,20],"introduct":48,"layer":[24,25,26,27,28,29,30,31,49,50],"layernorm":26,"mask":[12,13],"model":[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],"multiheadattent":27,"my":49,"positionalencod":28,"project_config":[2,3],"projectionlay":29,"quick":47,"residualconnect":30,"scale":49,"sci_tf":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],"sciform":47,"select":47,"take":49,"tf_infer":[17,18],"tf_util":[44,45],"tf_visual":[19,20],"tfinfer":18,"tfvisual":20,"tokenembed":31,"trainer":[38,39,40,41,42],"transform":[47,50],"transformer_model":[32,33,34,35,36,37],"transformer_train":[39,40],"transformer_valid":[41,42],"transformermodel":37,"transformertrain":40,"transformervalid":42,"two_language_data_set":[14,15],"twolanguagesdataset":15,"type":50,"util":[43,44,45],"what":49}})