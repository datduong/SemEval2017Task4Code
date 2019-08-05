
## run bert to classify sentiment score high/low


server='/local/datdb'

work_dir=$server/'YelpReviewUserEmb/Data'

bert_model=$work_dir/'TextData/BertFineTune' # use the full mask + nextSentence to innit
pregenerated_data=$bert_model # use the data of full mask + nextSentence to innit
bert_output_dir=$bert_model/'ClassifyScore'
mkdir $bert_output_dir

data_dir=$work_dir/'TextData'

result_folder=$bert_output_dir/'run1'
mkdir $result_folder

conda activate tensorflow_gpuenv
cd $server/YelpReviewUserEmb

CUDA_VISIBLE_DEVICES=4 python3 $server/YelpReviewUserEmb/BERT/sentiment/do_model.py --fp16 --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 24 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --num_train_epochs_entailment 100 --use_cuda > $result_folder/train.log


#### **** DO TESTING, AND WRITE OUT VECTORS 


server='/local/datdb'

work_dir=$server/'YelpReviewUserEmb/Data'

bert_model=$work_dir/'TextData/BertFineTune' # use the full mask + nextSentence to innit
pregenerated_data=$bert_model # use the data of full mask + nextSentence to innit
bert_output_dir=$bert_model/'ClassifyScore'
mkdir $bert_output_dir

data_dir=$work_dir/'TextData'

result_folder=$bert_output_dir/'run1'
mkdir $result_folder

# file_extract_vec_in=$work_dir/'yelpchallenge_09/user_tag_append_bert.txt'
# file_extract_vec_out=$result_folder/'user_tag_append_bert_vector.txt'


file_extract_vec_in=$work_dir/'TextData/concept_word_bert_input.txt'
file_extract_vec_out=$result_folder/'concept_tag_append_bert_vector.txt'


conda activate tensorflow_gpuenv
cd $server/YelpReviewUserEmb

CUDA_VISIBLE_DEVICES=4 python3 $server/YelpReviewUserEmb/BERT/sentiment/do_model.py --main_dir $work_dir --qnli_dir $data_dir --batch_size_label 8 --bert_model $bert_model --pregenerated_data $pregenerated_data --bert_output_dir $bert_output_dir --result_folder $result_folder --not_train --file_extract_vec_in $file_extract_vec_in --file_extract_vec_out $file_extract_vec_out --use_cuda > $result_folder/test.log



