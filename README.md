# Model Personalizatin with Static and Dynamic Patients' Data

## Data from MIMIC-Extract
* run [./data/data_extraction_mimic_extract.py](data_extraction_mimic_extract.py) to prepare the bp data for experiments. Put the output under 'DATA_DIR/data/bp/' or change the directory in [data_utils](./data/data_utils.py)
* Note that for this step you need to have access to MIMIC-III
* The models for generating the amplitude and phase for the SX dataset can be found under [./data/sx/](./data/sx/)
* The generic models for each dataset and setting are included under their directory

## Experiments
* run [w_static.py](w_static.py) for the w/ static models, [wo_static.py](wo_static.py) for the w/o static models under './experiments/DATASET_NAME' to get the resulls for ml models and tl models
* run [eval_train.py](eval_train.py) under the same directory to generate the task-specific models for the training set
* Correspondingly, run [reconst_gen.py](reconst_gen.py) under each directory to construct models based on static attributes using td-maml


## Citation
```
@INPROCEEDINGS{youssef-etal-2022-personalization,
  author={Youssef, Paul and Schlötterer, Jörg and Imangaliyev, Sultan and Seifert, Christin},
  booktitle={2022 IEEE International Conference on Data Mining Workshops (ICDMW)}, 
  title={Model Personalization with Static and Dynamic Patients' Data}, 
  year={2022},
  pages={324-333},
  doi={10.1109/ICDMW58026.2022.00051}}
```
