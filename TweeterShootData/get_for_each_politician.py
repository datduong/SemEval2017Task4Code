
# https://github.com/twintproject/twint/wiki/Scraping-functions

import re,os,sys,pickle,math
import pandas as pd
import twint


def submitJobs (name):
  c = twint.Config()
  c.Username = name # "TiffanyGalik" # "EleanorNorton"
  c.User_full = False
  # c.Store_csv = True
  twint.run.Followers(c)


if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] )



# #!/bin/bash
# . /u/local/Modules/default/init/modules.sh
# module load python/3.7.2
# cd /u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/TweeterShootData
# python3 get_for_each_politician.py AGBecerra > /u/scratch/d/datduong/framing-twitter/data/PoliticianFollower/AGBecerra_followers.txt



