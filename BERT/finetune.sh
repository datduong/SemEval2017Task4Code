


## generate data for fine tune LM model


#!/bin/bash
## make data for training (about 30 mins) on hoffman
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
server='/u/flashscratch/d/datduong'
data_dir=$server/'SemEval2017Task4/4B-English'
output_dir=$data_dir/'BertFineTuneAddGamerGate'
mkdir $output_dir

# train_file=$data_dir/task4B_bert_pretrain_file.txt

train_file=$data_dir/task4B_addgamergate_bert_pretrain_file.txt

cd $server/pytorch-transformers/examples/lm_finetuning
python3 pregenerate_training_data.py --train_corpus $train_file --bert_model bert-base-cased --output_dir $output_dir --epochs_to_generate 10 --max_seq_len 512



## LM tune the data
conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'SemEval2017Task4/4B-English'
output_dir=$data_dir/'BertFineTuneAddGamerGate'
bert_model='/local/datdb/BERTPretrainedModel/cased_L-12_H-768_A-12/'
mkdir $output_dir

cd /local/datdb/pytorch-transformers/examples/lm_finetuning

CUDA_VISIBLE_DEVICES=1 /local/datdb/anaconda3/envs/tensorflow_gpuenv/bin/python -u finetune_on_pregenerated.py --pregenerated_data $output_dir --bert_model $bert_model --output_dir $output_dir --epochs 10 --train_batch_size 6 

#--fp16

