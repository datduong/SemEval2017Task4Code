## MAKE TRAIN DEV TEST 
module load python/3.7.2
cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/Data
main_dir='/u/scratch/d/datduong/GamergateTweet/'
fout='GamergateTweetTextUserData'
do_filter_test_label='0'
to_skip='none'
topic_to_test_file='none'
where_save='SplitData'
base_name='NotMask'

python3 make_fold.py $main_dir $fout'.txt' $to_skip $do_filter_test_label $topic_to_test_file $where_save $base_name



