


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

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

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)


config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForSequenceClassification(config)
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
input_ids = torch.cat((input_ids,input_ids),0)

model.bert(input_ids)


embedding_output = model.bert.embeddings(input_ids)
mask = torch.ones(2,6)
head_mask = [None] * config.num_hidden_layers

model.bert.encoder.output_hidden_states = True
encoder_outputs = model.bert.encoder(embedding_output,None,head_mask)



for k in range(len(encoder_outputs[1])): 
  print ( encoder_outputs[1][k].shape ) 




