
#!/bin/bash


cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/BERT

data_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English'
out_dir=$data_dir/'W2vEmbTweet'
mkdir $out_dir

## train w2v on tweets

# . /u/local/Modules/default/init/modules.sh
# module load python/3.7.2

/u/home/d/datduong/anaconda2/bin/python trainW2vModel.py $data_dir $out_dir W2vEmbTweet 100 

