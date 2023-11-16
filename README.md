# Observatory
Codebase of paper [Observatory: Characterizing Embeddings of Relational Tables](https://arxiv.org/abs/2310.07736).

## Citing This Repository
If you find this repository useful for your work, please cite the following BibTeX:

```bibtex
@article{cong2023observatory,
  author  = {Tianji Cong and
             Madelon Hulsebos and
             Zhenjie Sun and
             Paul Groth and
             H. V. Jagadish},
  title   = {Observatory: Characterizing Embeddings of Relational Tables},
  journal = {Proc. {VLDB} Endow.},
  year    = {2024}
}
```

```bibtex
@inproceedings{cong2023observatorylibrary,
  author    = {Cong, Tianji and
               Sun, Zhenjie and
               Groth, Paul and
               Jagadish, H. V. and
               Hulsebos, Madelon},
  title     = {Introducing the Observatory Library for End-to-End Table Embedding Inference},
  booktitle = {The 2nd Table Representation Learning Workshop at NeurIPS 2023},
  publisher = {https://table-representation-learning.github.io},
  year      = {2023}
}
```
 
## Environment Setup
Assume using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) for Python package management on Linux machines. 

1. Clone this repo in your working directory (the ```--recursive``` flag is necessary to pull a dependent repo like [TURL](https://github.com/sunlab-osu/TURL) as a submodule):

    ```git clone <Observatory repo url> --recursive```

    ```cd observatory```

2. Create and activate the development environment:

    ```conda env create -f environment.yml ```

    ```conda activate observatory```

    Note that TaBERT was developed earlier and has many dependencies that are not compatible with other models. We create a separate environment for running TaBERT.

    ```conda env create -f tabert.yml```

    ```conda activate tabert```

3. Import Observatory and other dependencies as editable packages to the conda environment

    ```conda develop <path to Observatory>```

    ```conda develop <path to DODUO>```
    
    ```conda develop <path to TURL>```

    ```conda develop <path to TaBERT>```

    e.g.,
    
    ```conda develop /home/congtj/observatory```

    ```conda develop /home/congtj/observatory/observatory/models/TURL```

## Models and Data
|  Model  	|           Checkpoint Identifier           	|                               Link                              	|
|---------	|-------------------------------------------	|-----------------------------------------------------------------	|
| BERT    	| "bert-base-uncased"                       	| https://huggingface.co/docs/transformers/model_doc/bert         	|
| RoBERTa 	| "roberta-base"                            	| https://huggingface.co/docs/transformers/model_doc/roberta      	|
| T5      	| "t5-base"                                 	| https://huggingface.co/docs/transformers/model_doc/t5           	|
| TAPAS   	| "google/tapas-base"                       	| https://huggingface.co/docs/transformers/model_doc/tapas        	|
| TaBERT  	| "TaBERT_Base_(K=3)"                       	| https://github.com/facebookresearch/TaBERT#pre-trained-models   	|
| TURL    	| "checkpoint/pretrained/pytorch_model.bin" 	| https://github.com/sunlab-osu/TURL#data                         	|
| DODUO   	| "wikitable"                               	| https://github.com/megagonlabs/doduo/tree/main#data-preparation 	|


| Dataset             	| Files                                        	| Links                                                                                                                  	| Evaluated Properties                                                 	|
|---------------------	|----------------------------------------------	|-----------------------------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------	|
| WikiTables          	| data/entity_vocab.txt data/test_tables.jsonl 	| https://github.com/sunlab-osu/TURL#data                                                                               	| Row Order Insignificance Column Order Insignificance Sample Fidelity 	|
| NextiaJD            	| XS, S, M, L                                  	| https://github.com/dtim-upc/NextiaJD/tree/master/experiments#setting                                                  	| Join Relationship                                                    	|
| Dr.Spider           	| data.tar.gz                                  	| https://github.com/awslabs/diagnostic-robustness-text-to-sql                                                          	| Perturbation Robustness                                              	|
| Spider              	| Spider Dataset spider_fd_artifact.zip        	| https://yale-lily.github.io/spider https://drive.google.com/file/d/1br0voV0l3yBMfEy9WM7Vja-cZX6HvvL1/view?usp=sharing 	| Functional Dependencies                                              	|
| WikiTables_Entities 	| entity_stability_queries.csv                 	| https://drive.google.com/file/d/1SM_AOpmFbh51IUTQuvI7YETLdjThgpSB/view?usp=sharing                                    	| Entity Stability                                                     	|
| SOTAB               	| sotab_data_type_datasets.zip                 	| https://drive.google.com/file/d/1K631KONGDVy2C2ViKcwSyMnWhr_kJdaK/view?usp=share_link                                 	| Heterogeneous Context                                                	|

## Evaluation
All property experiments are under `observatory/properties`. To evaluate a model for a property, change all paths to local ones in the corresponding shell script and run

For models from HuggingFace (specify the model identifier in the script): e.g.,

    ./observatory/properties/Row_Order_Insignificance/hugging_face.sh
    
For other models: e.g.,

    ./observatory/properties/Sample_Fidelity/doduo.sh

## Side Notes
To evaluate the DODUO model on GPU, please substitute ```observatory/models/DODUO/doduo/doduo.py``` with ```observatory/models/doduo.py```.
