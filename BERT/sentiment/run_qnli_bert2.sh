


## MAKE TRAIN DEV TEST 
module load python/3.7.2
cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/Data
main_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
fout='task4B_bert_sentiment_nonan_user'
do_filter_test_label='1'
to_skip='none'
topic_to_test_file='topic_to_test_3_7'
where_save='BertSentimentNoNanUser'

for base_name in user_data ; do 
  python3 make_fold.py $main_dir $fout'_mask_'$base_name'.txt' $to_skip $do_filter_test_label $topic_to_test_file $where_save 'mask_'$base_name
done 



conda activate tensorflow_gpuenv

for folder_type in user_data ; do 

  data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentNoNanUser/mask_'$folder_type # full_data_mask
  output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentNoNanUser/mask_'$folder_type
  mkdir $output_dir
  model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
  config_name=$model_name_or_path/'bert_config.json'
  tokenizer_name='bert-base-cased'

  cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment
  CUDA_VISIBLE_DEVICES=3 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 10 --do_train --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 --logging_steps 1000 --save_steps 1000 > $output_dir/track.log

done 


### TESTING 

keep_gender keep_name keep_loc keep_desc keep_name_mask_text keep_desc_mask_text keep_loc_mask_text mask_user_data keep_gender_mask_text

conda activate tensorflow_gpuenv 
for folder in keep_desc ; do 

  fold_where_test_file='/local/datdb/SemEval2017Task4/4B-English/BertSentimentNoNanUser'
  data_dir=$fold_where_test_file/$folder
  output_dir=$fold_where_test_file/$folder
  mkdir $output_dir
  model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
  config_name=$model_name_or_path/'bert_config.json' ## doesnt matter, once we load the model, this will be override
  tokenizer_name='bert-base-cased'

  model_name_or_path=$output_dir/checkpoint-5000 ## so that we load in newer model, this will override the init finetune
  
  cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment

  for test_data_type in 'test' ; do
    test_file=$fold_where_test_file'/'$folder'/'$test_data_type'.tsv'
    CUDA_VISIBLE_DEVICES=1 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 20 --do_eval --test_file $test_file --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 > $output_dir/$folder'_'$test_data_type.log
  done

done


### TEST ON GAMER GATE

conda activate tensorflow_gpuenv 
for folder in Base ; do 

  fold_where_test_file='/local/datdb/SemEval2017Task4/4B-English/BertSentimentNoNanUser'
  data_dir=$fold_where_test_file/$folder
  output_dir=$fold_where_test_file/$folder
  mkdir $output_dir
  model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
  config_name=$model_name_or_path/'bert_config.json' ## doesnt matter, once we load the model, this will be override
  tokenizer_name='bert-base-cased'

  model_name_or_path=$output_dir ## so that we load in newer model, this will override the init finetune
  
  cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment

  for test_data_type in 'test' ; do
    test_file='/local/datdb/GamergateTweet/SplitData/NotMask/test.tsv' # $fold_where_test_file'/'$folder'/'$test_data_type'.tsv'
    CUDA_VISIBLE_DEVICES=1 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 20 --do_eval --test_file $test_file --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 > $output_dir/$folder'_gamergate_'$test_data_type.log
  done

done



### TEST ON SET OF TOPICS. WE HAVE TO MASK OUT TEXT IN THIS CASE
conda activate tensorflow_gpuenv 
where_test_file='/local/datdb/SemEval2017Task4/4B-English/PredictTopicNoNanUser'
main_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentNoNanUser'

for folder in mask_text ; do 

  data_dir=$main_dir/$folder
  output_dir=$main_dir/$folder
  mkdir $output_dir
  model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
  config_name=$model_name_or_path/'bert_config.json' ## doesnt matter, once we load the model, this will be override
  tokenizer_name='bert-base-cased'

  model_name_or_path=$output_dir ## so that we load in newer model, this will override the init finetune
  output_save=$output_dir/'PredictTopic'
  mkdir $output_save
  
  cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment

  for test_data_type in 'iran' 'gay' 'islam' 'muslims' 'christians' 'red_sox' 'rolling_stones' 'miss_usa' 'twilight' ; do
    test_file=$where_test_file'/'$test_data_type'.txt'
    CUDA_VISIBLE_DEVICES=1 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 20 --do_eval --test_file $test_file --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 --write_prob $output_save/$test_data_type'.tsv' > $output_save/$test_data_type'.log'
  done

done
