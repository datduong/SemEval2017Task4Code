



import os,sys,re,pickle
import numpy as np
import pandas as pd
## extract only user in test set, see what is accuracy 

main_dir = '/local/datdb/SemEval2017Task4/4B-English'
os.chdir(main_dir)

topic_list = pd.read_csv("/local/datdb/SemEval2017Task4/4B-English/topic_to_test_3_7.txt",sep="\t",header=None)
topic_list = list (topic_list[0])
topic_list = ["test_user_only_"+re.sub(" ","_",top) for top in topic_list] 

script = """
conda activate tensorflow_gpuenv
## *** do testing 
for folder in full_data_mask_text ; do 

  data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel37/'$folder
  output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel37/'$folder
  mkdir $output_dir
  model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
  config_name=$model_name_or_path/'bert_config.json'
  tokenizer_name='bert-base-cased'

  model_name_or_path=$output_dir ## so that we load in newer model
  fold_where_test_file='/local/datdb/SemEval2017Task4/4B-English/BertSentimentFilterTestLabel37'
  output_dir_log=$output_dir/'by_topic'
  mkdir $output_dir_log

  cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment

  # full_data_mask full_data_mask_name_description_location_user_gender full_data_mask_text full_data_mask_description

  for folder_of_test_file in full_data_mask_text ; do 
    output_dir_log_folder=$output_dir/'by_topic'/$folder_of_test_file
    mkdir $output_dir_log_folder
    for test_data_type in FILEHERE ; do 
      test_file=$fold_where_test_file'/'$folder_of_test_file'/'$test_data_type'.tsv'
      CUDA_VISIBLE_DEVICES=4 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 20 --do_eval --test_file $test_file --max_seq_length 512 --overwrite_output_dir --evaluate_during_training --num_segment_type 6 --learning_rate 0.00001 --fp16 > $output_dir_log_folder/$test_data_type.log
    done
  done
done

"""

script = re.sub("FILEHERE", " ".join(topic_list), script)
fout=open('script_test_by_topic3.sh','w')
fout.write(script)
fout.close()
