

## make data for training QNLI style
module load python/3.7.2
cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/Data
main_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
fout='task4B_bert_sentiment_file_full'
to_skip='none'
python3 make_bert_sentimental_data.py $main_dir $fout $to_skip
python3 make_fold.py $main_dir $fout $to_skip



## mask out some information 
module load python/3.7.2
cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/Data
main_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
for to_skip in name description location user_gender ; do 
  fout='task4B_bert_sentiment_file_full'
  python3 make_bert_sentimental_data.py $main_dir $fout $to_skip
  python3 make_fold.py $main_dir $fout'_'$to_skip'.txt' $to_skip
done



## can we load a model with more than 2 id types ??? yes ... we can trick the model 



## run entailment based on BERT. using QNLI as template input 

conda activate tensorflow_gpuenv
cd /local/datdb/SemEval2017Task4/SemEval2017Task4Code/BERT/sentiment

data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/full_data_mask_type/'
output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/full_data_mask_type/'
model_name_or_path='/local/datdb/SemEval2017Task4/4B-English/BertFineTune/' ## load fine tune with just 2 tokens 
config_name=$model_name_or_path/'bert_config.json'
tokenizer_name='bert-base-cased'

CUDA_VISIBLE_DEVICES=4 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 10 --do_train --max_seq_length 512 --overwrite_output_dir --fp16 --evaluate_during_training --num_segment_type 6 > $output_dir/track.log




## *** EVALUATION 

conda activate tensorflow_gpuenv
cd /local/datdb/pytorch-transformers/examples

data_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/notweet_fold_1/'
output_dir='/local/datdb/SemEval2017Task4/4B-English/BertSentiment/notweet_fold_1/'
model_name_or_path=$output_dir
config_name=$model_name_or_path/'bert_config.json'
tokenizer_name='bert-base-cased'

## eval uses dev.tsv, so we "trick" the input by naming test-->dev, and set dev-->dev_original 
CUDA_VISIBLE_DEVICES=4 python3 -u run_glue.py --data_dir $data_dir --model_type bert --model_name_or_path $model_name_or_path --task_name qnli --output_dir $output_dir --config_name $config_name --tokenizer_name $tokenizer_name --num_train_epochs 10 --do_eval --max_seq_length 512 --overwrite_output_dir > $output_dir/track.log



## *** GET WORD VECTORS

## create filles with words to extract vectors from 
cd $SCRATCH/SemEval2017Task4/SemEval2017Task4Code/  
python3 BERT/sentiment/word_to_write.py 


##
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


## convert into form for @hist_words 
python3 BERT/sentiment/convert_to_np.py

