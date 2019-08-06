

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os, sys, pickle
import random
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

sys.path.append("/local/datdb/pytorch-transformers/examples")

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}


sys.path.append("/local/datdb/SemEval2017Task4/SemEval2017Task4Code/")
import BERT.sentiment.vector_extractor as vector_extractor
import BERT.sentiment.arg_input as arg_input
args = arg_input.get_args()


## extract some layer in bert to represent the user/keyword vectors


## follow the same technique https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py
args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=2, finetuning_task=args.task_name) ## hard code @num_labels, because we know it's entailment style

tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

ModelWriter = vector_extractor.Extractor2ndLast(model) ## probably not the smartest way ...
ModelWriter.cuda()


## !!!! load label data ... label is one word like "feminist"
label_desc_loader = vector_extractor.make_loader (args,args.word_vector_input,tokenizer,64)

label_name = pd.read_csv(args.word_vector_input,header=None)
label_name = list ( label_name[0] )
label_name = [ re.sub(" ","_",lab) for lab in label_name ] ## because gensim uses space delim

print ('see some input words to get vector for ...')
print (label_name[0:10])

label_emb = ModelWriter.write_vector (label_desc_loader,args.word_vector_output,label_name)



