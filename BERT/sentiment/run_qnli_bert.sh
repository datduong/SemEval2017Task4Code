
## run entailment based on BERT. using QNLI as template input 

conda activate tensorflow_gpuenv
cd /local/datdb/pytorch-transformers/examples

data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1/'
output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1/'
model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/'
config_name=$model_name_or_path/'bert_config.json'
tokenizer_name='bert-base-cased'


CUDA_VISIBLE_DEVICES=4 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 10 --do_train --max_seq_length 512 --overwrite_output_dir > $output_dir/track.log




## *** EVALUTATION 

conda activate tensorflow_gpuenv
cd /local/datdb/pytorch-transformers/examples

data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1/'
output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1/'
model_name_or_path=$output_dir
config_name=$model_name_or_path/'bert_config.json'
tokenizer_name='bert-base-cased'

## eval uses dev.tsv, so we "trick" the input by naming test-->dev, and set dev-->dev_original 
CUDA_VISIBLE_DEVICES=4 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 10 --do_eval --max_seq_length 512 --overwrite_output_dir > $output_dir/track.log



## *** GET WORD VECTORS

conda activate tensorflow_gpuenv
cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/  # BERT/sentiment

data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1/'
output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1/'
model_name_or_path=$output_dir
config_name=$model_name_or_path/'bert_config.json'
tokenizer_name='bert-base-cased'

word_vector_input='/local/datdb/SemEval2017Task4/4B-English/word_to_get_vec.txt'
word_vector_output=$output_dir/'word_vector.txt'

## eval uses dev.tsv, so we "trick" the input by naming test-->dev, and set dev-->dev_original 
CUDA_VISIBLE_DEVICES=4 python3 -u BERT/sentiment/write_vector.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --max_seq_length 512 --overwrite_output_dir --word_vector_input $word_vector_input --word_vector_output $word_vector_output > $output_dir/track.log



