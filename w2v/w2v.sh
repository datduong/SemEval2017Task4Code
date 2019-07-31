
#!/bin/bash

cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/w2v

data_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English'

dim=300
for choice in male FEMALE ; do 
  file_wanted='SemEval2017-task4-dev.subtask-BD.english.'$choice'.w2v.txt'
  fout_name='W2vEmbTweetTopic'$choice
  out_dir=$data_dir/$fout_name
  mkdir $out_dir

  /u/home/d/datduong/anaconda2/bin/python trainW2vModel.py $data_dir $out_dir $file_wanted $fout_name $dim > $choice.log

  python3 w2v_to_text.py $fout_name $dim

done


#!/bin/bash

cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/w2v

data_dir='/u/scratch/d/datduong/SemEval2017Task4/4B-English'

dim=300
for choice in male FEMALE ; do 

  file_wanted='SemEval2017-task4-dev.subtask-BD.english.user.'$choice'.w2v.txt'
  fout_name='W2vEmbTweetUser'$choice
  out_dir=$data_dir/$fout_name
  mkdir $out_dir

  /u/home/d/datduong/anaconda2/bin/python trainW2vModel.py $data_dir $out_dir $file_wanted $fout_name $dim > $choice.log

  python3 w2v_to_text.py $fout_name $dim

done 



