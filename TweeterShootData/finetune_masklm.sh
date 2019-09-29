

#!/bin/bash
## make data for training (about 30 mins) on hoffman
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
server='/u/flashscratch/d/datduong'
data_dir=$server/'TweetShootData2018'
output_dir=$data_dir/'pretrain_data'
# mkdir $output_dir

# train_file=$data_dir/task4B_bert_pretrain_file.txt

train_file=$data_dir/tweet_pretrain.txt

cd $server/pytorch-transformers/examples/lm_finetuning
python3 pregenerate_training_data.py --train_corpus $train_file --bert_model bert-base-cased --output_dir $output_dir --epochs_to_generate 10 --max_seq_len 512



## LM tune the data
conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'TweetShootData2018/pretrain_data'
output_dir=$data_dir/'pretrainMLM'
bert_model='/local/datdb/BERTPretrainedModel/cased_L-12_H-768_A-12/'
mkdir $output_dir

cd /local/datdb/pytorch-transformers/examples/lm_finetuning
CUDA_VISIBLE_DEVICES=1 python3 -u finetune_on_pregenerated.py --pregenerated_data $data_dir --bert_model $bert_model --output_dir $output_dir --epochs 10 --train_batch_size 6


## !!!

## we can run the newest maskLM code ?? this will only do maskLM
## problem with booleans

# cd $server/BertGOAnnotation/finetune/
cd $server/transformers/examples


train_masklm_data=$data_dir/'tweet_pretrain_25percent_test.txt'
eval_masklm_data=$data_dir/'tweet_pretrain_25percent_test.txt'

CUDA_VISIBLE_DEVICES=1 python3 -u run_lm_finetuning.py --block_size 512 --mlm --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 10 --per_gpu_train_batch_size 8 --do_train --model_type bert --overwrite_output_dir --save_steps 500 --logging_steps 500 --evaluate_during_training --eval_data_file $eval_masklm_data --fp16

