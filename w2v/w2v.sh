
#!/bin/bash


cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/w2v

data_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English'

choice='all'
file_wanted='SemEval2017-task4-dev.subtask-BD.english.all.w2v.txt'
out_dir=$data_dir/'W2vEmbTweet'$choice
mkdir $out_dir


## train w2v on tweets

# . /u/local/Modules/default/init/modules.sh
# module load python/3.7.2

/u/home/d/datduong/anaconda2/bin/python trainW2vModel.py $data_dir $out_dir $file_wanted W2vEmbTweet$choice 100 > $choice.log

