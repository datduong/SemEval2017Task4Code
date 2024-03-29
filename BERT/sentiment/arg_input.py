


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

logger = logging.getLogger(__name__)


from argparse import ArgumentParser
from pathlib import Path

# start_near_true_label = False ## global


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}


def get_args():
  parser = ArgumentParser(
      description='encode label descriptions into vectors')

  ## Required parameters
  parser.add_argument("--word_vector_input", default=None, type=str,
                      help="File of words to be converted into vectors")
  parser.add_argument("--word_vector_output", default=None, type=str,
                      help="Output words and their vectors.")
  parser.add_argument("--data_dir", default=None, type=str, required=True,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--model_type", default=None, type=str, required=True,
                      help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                      help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
  parser.add_argument("--task_name", default=None, type=str, required=True,
                      help="The name of the task to train selected in the list.")
  parser.add_argument("--output_dir", default=None, type=str, required=True,
                      help="The output directory where the model predictions and checkpoints will be written.")

  ## Other parameters
  parser.add_argument("--config_name", default="", type=str,
                      help="Pretrained config name or path if not the same as model_name")
  parser.add_argument("--tokenizer_name", default="", type=str,
                      help="Pretrained tokenizer name or path if not the same as model_name")
  parser.add_argument("--cache_dir", default="", type=str,
                      help="Where do you want to store the pre-trained models downloaded from s3")
  parser.add_argument("--max_seq_length", default=128, type=int,
                      help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
  parser.add_argument("--do_train", action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval", action='store_true',
                      help="Whether to run eval on the dev set.")
  parser.add_argument("--evaluate_during_training", action='store_true',
                      help="Rul evaluation during training at each logging step.")
  parser.add_argument("--do_lower_case", action='store_true',
                      help="Set this flag if you are using an uncased model.")

  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                      help="Batch size per GPU/CPU for training.")
  parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                      help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument("--learning_rate", default=5e-5, type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float,
                      help="Weight deay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                      help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float,
                      help="Max gradient norm.")
  parser.add_argument("--num_train_epochs", default=3.0, type=float,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--max_steps", default=-1, type=int,
                      help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
  parser.add_argument("--warmup_steps", default=0, type=int,
                      help="Linear warmup over warmup_steps.")

  parser.add_argument('--logging_steps', type=int, default=50,
                      help="Log every X updates steps.")
  parser.add_argument('--save_steps', type=int, default=50,
                      help="Save checkpoint every X updates steps.")
  parser.add_argument("--eval_all_checkpoints", action='store_true',
                      help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
  parser.add_argument("--no_cuda", action='store_true',
                      help="Avoid using CUDA when available")
  parser.add_argument('--overwrite_output_dir', action='store_true',
                      help="Overwrite the content of the output directory")
  parser.add_argument('--overwrite_cache', action='store_true',
                      help="Overwrite the cached training and evaluation sets")
  parser.add_argument('--seed', type=int, default=42,
                      help="random seed for initialization")

  parser.add_argument('--fp16', action='store_true',
                      help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
  parser.add_argument('--fp16_opt_level', type=str, default='O1',
                      help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
  parser.add_argument("--local_rank", type=int, default=-1,
                      help="For distributed training: local_rank")
  parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
  parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
  
  args = parser.parse_args()
  return args


