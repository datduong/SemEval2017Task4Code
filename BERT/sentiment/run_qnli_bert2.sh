


## keep**** something + tweet task4B_bert_sentiment_add_gamergate ...
module load python/3.7.2
cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/Data
main_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
fout='task4B_bert_sentiment_nonan_user'
do_filter_test_label='1'
to_skip='none'
topic_to_test_file='topic_to_test_3_7'
where_save='BertSentimentNoNanUser'

for base_name in name desc loc gender ; do 
  python3 make_fold.py $main_dir $fout'_keep_'$base_name'_mask_text.txt' $to_skip $do_filter_test_label $topic_to_test_file $where_save 'keep_'$base_name'_mask_text'
done 



conda activate tensorflow_gpuenv

for folder_type in name desc loc gender ; do 

  data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentNoNanUser/keep_'$folder_type'_mask_text' # full_data_mask
  output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentNoNanUser/keep_'$folder_type'_mask_text'
  mkdir $output_dir
  model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
  config_name=$model_name_or_path/'bert_config.json'
  tokenizer_name='bert-base-cased'

  cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment
  CUDA_VISIBLE_DEVICES=6 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 10 --do_train --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 --logging_steps 1000 --save_steps 1000 > $output_dir/track.log

done 

