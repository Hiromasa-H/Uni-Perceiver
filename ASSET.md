
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
python3 -m main.py --num-gpus 1 --config-file /home/hhiromasa/code/Uni-Perceiver/configs/BERT_L12_H768_experiments/finetuning/GLUE_finetuning_experiments/GLUE_QQP_mlm_finetune.yaml  OUTPUT_DIR /home/hhiromasa/code/Uni-Perceiver/asset/FT_glue MODEL.WEIGHTS /home/hhiromasa/code/Uni-Perceiver/weights/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth
```

## ImageNet1K Fine-Tuning


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