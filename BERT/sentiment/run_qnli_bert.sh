
## run entailment based on BERT. using QNLI as template input 

conda activate tensorflow_gpuenv
cd /local/datdb/pytorch-transformers/examples

data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1/'
output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1/'
model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/'
config_name=$model_name_or_path/'bert_config.json'
tokenizer_name='bert-base-cased'


CUDA_VISIBLE_DEVICES=4 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 10 --do_train --max_seq_length 512 --overwrite_output_dir > $output_dir/track.log


