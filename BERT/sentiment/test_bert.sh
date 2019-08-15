
conda activate tensorflow_gpuenv

for folder in full_data_mask full_data_mask_name_description_location_user_gender full_data_mask_text full_data_mask_description ; do 

data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel'/$folder
output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel'/$folder
mkdir $output_dir
model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
config_name=$model_name_or_path/'bert_config.json'
tokenizer_name='bert-base-cased'

cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment
model_name_or_path=$output_dir ## so that we load in newer model
fold_where_test_file='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel'
output_dir_log=$output_dir/'by_topic'
mkdir $output_dir_log

for test_data_type in test_user_only ; do
  test_file=$fold_where_test_file'/'$folder'/'$test_data_type'.tsv'
  CUDA_VISIBLE_DEVICES=1 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 20 --do_eval --test_file $test_file --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 > $output_dir_log/test_$folder'_'$test_data_type.log
done

done


full_data_mask full_data_mask_description full_data_mask_name_description_location_user_gender

conda activate tensorflow_gpuenv
for folder in full_data_mask_name_description_location full_data_mask_name_description_user_gender ; do 

  data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel37'/$folder
  output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel37'/$folder
  mkdir $output_dir
  model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
  config_name=$model_name_or_path/'bert_config.json'
  tokenizer_name='bert-base-cased'

  model_name_or_path=$output_dir ## so that we load in newer model
  fold_where_test_file='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel37'
  output_dir_log=$output_dir/'by_topic'
  mkdir $output_dir_log

  cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment

  # full_data_mask_text full_data_mask_name_location_user_gender full_data_mask full_data_mask_name full_data_mask_user_gender full_data_mask_location full_data_mask_text full_data_mask_description full_data_mask_description_name full_data_mask_description_location full_data_mask_description_user_gender full_data_mask_name_description_location_user_gender
  for folder_of_test_file in full_data_mask_name_description_location full_data_mask_name_description_user_gender ; do 
    for test_data_type in test_user_only ; do
      test_file=$fold_where_test_file'/'$folder_of_test_file'/'$test_data_type'.tsv'
      CUDA_VISIBLE_DEVICES=3 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 20 --do_eval --test_file $test_file --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 > $output_dir_log/$folder_of_test_file'_'$test_data_type.log
    done
  done

done



###

conda activate tensorflow_gpuenv

for folder in full_data_mask_text full_data_mask_name_location_user_gender full_data_mask_name_description_location full_data_mask_name_description_user_gender full_data_mask_name_description_location_user_gender ; do 

  data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel37'/$folder
  output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel37'/$folder
  mkdir $output_dir
  model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
  config_name=$model_name_or_path/'bert_config.json'
  tokenizer_name='bert-base-cased'

  cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment
  model_name_or_path=$output_dir ## so that we load in newer model
  fold_where_test_file='/local/datdb/GamergateTweet'
  output_dir_log=$output_dir/'gamergate'
  mkdir $output_dir_log

  test_file=$fold_where_test_file'/GamergateTweetTextUserData.tsv'

  CUDA_VISIBLE_DEVICES=1 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 20 --do_eval --test_file $test_file --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 > $output_dir_log/$folder'.log'


done

