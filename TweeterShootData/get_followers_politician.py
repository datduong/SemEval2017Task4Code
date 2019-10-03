

# https://github.com/twintproject/twint/wiki/Scraping-functions

import re,os,sys,pickle,math
import pandas as pd 

# script = """
# #!/bin/bash
# . /u/local/Modules/default/init/modules.sh
# module load python/3.7.2
# /u/home/d/datduong/.local/bin/twint -u username --followers > /u/scratch/d/datduong/framing-twitter/data/PoliticianFollower/username_followers.txt
# """


script = """
#!/bin/bash
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/TweeterShootData
# python3 get_for_each_politician.py username > /u/scratch/d/datduong/framing-twitter/data/PoliticianFollower/username_followers.txt

/u/home/d/datduong/.local/bin/twarc followers username > /u/scratch/d/datduong/framing-twitter/data/PoliticianFollower/username_followers.txt

"""


os.chdir('/u/scratch/d/datduong/tempTweetDownloadUser/')
politician_list = pd.read_csv("/u/scratch/d/datduong/framing-twitter/data/input/politician.txt",header=None) 
for counter,p in enumerate(list( politician_list[0] )) :
  script2 = re.sub('username',p,script)
  fout=open("run"+str(counter)+".sh","w")
  fout.write(script2)
  fout.close() 


###

output = '/u/scratch/d/datduong/tempTweetDownloadUser/'
size = 20
counter = int (math.ceil(counter*1.0/size)*size )

script = '#!/bin/bash\n#$ -cwd\n#$ -o /u/flashscratch/d/datduong/test.$JOB_ID.out\n#$ -j y\n#$ -t 1-UPPER:SIZE\nfor i in `seq 0 ADDON`; do\n\tmy_task_id=$((SGE_TASK_ID + i))\n\toutputrun$my_task_id.sh\ndone\n'

script = re.sub('output',output,script) 
script = re.sub('UPPER',str(counter),script)
script = re.sub('SIZE',str(size),script)
script = re.sub('ADDON',str(size-1),script) 
	
fout = open (output+"submitJobs.sh","w")
fout.write(script)
fout.close()

os.system ("chmod 777 -R "+output)
# os.system ("qsub -l h_data=4G,highp,h_rt=30:50:50 -pe shared 4 " + output+"submitJobs.sh" ) 


