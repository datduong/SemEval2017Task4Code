

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



