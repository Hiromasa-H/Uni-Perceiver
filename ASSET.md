
## Fine-tuning Tutorial
```
sh run.sh configs/BERT_L12_H768_experiments/finetuning/in1k_training.yaml in1k-ft 16 partitionname MODEL.WEIGHTS work_dirs/pretrained_models/uni-perceiver-base-L12-H768-224size-pretrained.pth
```

## GLUE QQP Fine-Tuning
<!--
This did not work. Seems like a command for slurm
```
sh run.sh configs/BERT_L12_H768_experiments/finetuning/in1k_training.yaml in1k-ft 16 partitionname MODEL.WEIGHTS /home/hhiromasa/code/Uni-Perceiver/weights/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth
```
-->
```
python3 main.py --num-gpus 1 --config-file /home/hhiromasa/code/Uni-Perceiver/configs/BERT_L12_H768_experiments/finetuning/GLUE_finetuning_experiments/GLUE_QQP_mlm_finetune.yaml  OUTPUT_DIR /home/hhiromasa/code/Uni-Perceiver/asset/FT_glue MODEL.WEIGHTS /home/hhiromasa/code/Uni-Perceiver/weights/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth
```

path: 
/home/hhiromasa/code/Uni-Perceiver/asset/FT_glue_fin/model_Epoch_10000_Iter_0009999.pth

## ImageNet1K Fine-Tuning
```
python3 main.py --num-gpus 1 --config-file /home/hhiromasa/code/Uni-Perceiver/configs/BERT_L12_H768_experiments/finetuning/in1k_training.yaml  OUTPUT_DIR /home/hhiromasa/code/Uni-Perceiver/asset/FT_in1k MODEL.WEIGHTS /home/hhiromasa/code/Uni-Perceiver/weights/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth
```

path:
/home/hhiromasa/code/Uni-Perceiver/asset/FT_in1k_fin/model_Epoch_01000_Iter_0000999.pth

## saving model weights as pickle
```
python3 model_to_pickle.py --num-gpus 1 --config-file /home/hhiromasa/code/Uni-Perceiver/configs/BERT_L12_H768_experiments/finetuning/in1k_training.yaml  OUTPUT_DIR /home/hhiromasa/code/Uni-Perceiver/asset/FT_in1k MODEL.WEIGHTS [weights to save]
```

## merging models
python3 model_merger.py

## infernce on in1k
```
python3 main.py --num-gpus 1 --config-file /home/hhiromasa/code/Uni-Perceiver/configs/BERT_L12_H768_experiments/zeroshot_config/in1k_training.yaml --eval-only OUTPUT_DIR /home/hhiromasa/code/Uni-Perceiver/asset/MM_in1k MODEL.WEIGHTS /home/hhiromasa/code/Uni-Perceiver/asset/FT_glue_fin/model_Epoch_10000_Iter_0009999.pth

/home/hhiromasa/code/Uni-Perceiver/asset/merged_model.pth 
```
## infernce on QQP
```
python3 main.py --num-gpus 1 --config-file /home/hhiromasa/code/Uni-Perceiver/configs/BERT_L12_H768_experiments/finetuning/GLUE_finetuning_experiments/GLUE_QQP_mlm_finetune.yaml --eval-only OUTPUT_DIR /home/hhiromasa/code/Uni-Perceiver/asset/MM MODEL.WEIGHTS /home/hhiromasa/code/Uni-Perceiver/asset/merged_model.pth 
```


### pip
```
mkl-service==2.4.0
pycocoevalcap==1.2
pycocotools==2.0.4
PyYAML==6.0
scikit-learn==1.0.2
scipy==1.7.3
sklearn==0.0
tensorboard==2.8.0


av==9.0.0
h5py==3.6.0
pyarrow==7.0.0
transformers==4.19.1
```

```
conda install conda-forge::pycocotools
conda install conda-forge::pyyaml
conda install anaconda::scikit-learn
conda install anaconda::scipy
conda install conda-forge::tensorboard
```

python3 main.py --num-gpus 1 --config-file /home/hhiromasa/code/Uni-Perceiver/configs/BERT_L12_H768_experiments/zeroshot_config/in1k_training.yaml --eval-only OUTPUT_DIR /home/hhiromasa/code/Uni-Perceiver/asset/MM_in1k MODEL.WEIGHTS /home/hhiromasa/code/Uni-Perceiver/asset/FT_in1k_fin/model_Epoch_01000_Iter_0000999.pth