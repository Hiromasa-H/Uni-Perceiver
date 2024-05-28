
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

## ImageNet1K Fine-Tuning
