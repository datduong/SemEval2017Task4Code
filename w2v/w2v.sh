


#!/bin/bash


cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/w2v

data_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English'

dim=300
choice='male'
file_wanted='SemEval2017-task4-dev.subtask-BD.english.male.w2v.txt'
fout_name='W2vEmbTweetTopic'$choice
out_dir=$data_dir/fout_name
mkdir $out_dir


## train w2v on tweets

# . /u/local/Modules/default/init/modules.sh
# module load python/3.7.2

/u/home/d/datduong/anaconda2/bin/python trainW2vModel.py $data_dir $out_dir $file_wanted $fout_name $dim > $choice.log

python3 w2v_to_text.py $fout_name $dim





