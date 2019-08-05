

from __future__ import absolute_import, division, print_function

import argparse,csv,logging,os,random,sys, pickle, gzip, json
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig, BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

sys.path.append("/local/datdb/pytorch-transformers/examples")
import utils_glue as utils_glue ### !!!! NOTICE, THE BERT CODE FROM HUGGINGFACE CHANGED NAME. THEY NOW HAVE THIS OBJECT @utils_glue

sys.path.append("/local/datdb/SemEval2017Task4/SemEval2017Task4Code")

import BERT.sentiment.arg_input as arg_input
args = arg_input.get_args()

import BERT.sentiment.data_loader as data_loader

import BERT.sentiment.model as encoder_model

MAX_SEQ_LEN = 512

# use BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True) # args.do_lower_case args.bert_tokenizer

name_add_on = ""
if args.fp16:
  name_add_on = "_fp16"


# get label-label entailment data
processor = data_loader.QnliProcessor()
label_list = processor.get_labels()
num_labels = len(label_list)


if not args.not_train : ## so we do not train 

  train_label_examples = processor.get_train_examples(args.qnli_dir) ## testing so use small data 1st

  # examples, label_list, max_seq_length, tokenizer, do_bert_tok=True
  train_label_features = data_loader.convert_examples_to_features(train_label_examples, label_list, MAX_SEQ_LEN, tokenizer)

  train_label_dataloader = data_loader.make_data_loader (train_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='random')
  print ('\ntrain_label_examples {}'.format(len(train_label_examples))) 


  """ get dev or test set  """

  # get label-label entailment data
  processor = data_loader.QnliProcessor()
  dev_label_examples = processor.get_dev_examples(args.qnli_dir)
  dev_label_features = data_loader.convert_examples_to_features(dev_label_examples, label_list, MAX_SEQ_LEN, tokenizer)
  dev_label_dataloader = data_loader.make_data_loader (dev_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='sequential')
  # torch.save( dev_label_dataloader, os.path.join( args.qnli_dir, "dev_label_dataloader"+name_add_on+".pytorch") )
  print ('\ndev_label_examples {}'.format(len(dev_label_examples))) 


## **** make model ****

# bert model

bert_config = BertConfig( os.path.join(args.bert_model,"bert_config.json") )

cache_dir = args.cache_dir if args.cache_dir else os.path.join(
  str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

bert_lm_sentence = BertForPreTraining.from_pretrained(args.bert_model,cache_dir=cache_dir)  # @num_labels is yes/no
if args.fp16:
  bert_lm_sentence.half() ## don't send to cuda, we will send to cuda with the joint model


# **** joint model ****

# make accessories variables for bert LM mask

samples_per_epoch = []
for i in range(int(args.num_train_epochs_bert)): ## how many bert epoch 
  epoch_file = args.pregenerated_data / f"epoch_{i}.json"
  metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
  if epoch_file.is_file() and metrics_file.is_file():
    metrics = json.loads(metrics_file.read_text())
    samples_per_epoch.append(metrics['num_training_examples'])
  else:
    if i == 0:
      exit("No training data was found!")
    print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.num_train_epochs_bert}).")
    print("This script will loop over the available data, but training diversity may be negatively impacted.")
    num_data_epochs = i
    break
else:
  num_data_epochs = int(args.num_train_epochs_bert)

if args.local_rank == -1 or args.no_cuda:
  device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
  n_gpu = torch.cuda.device_count()
else:
  torch.cuda.set_device(args.local_rank)
  device = torch.device("cuda", args.local_rank)
  n_gpu = 1
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')

logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

args.batch_size_bert = args.batch_size_bert // args.gradient_accumulation_steps

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.bert_output_dir.is_dir() and list(args.bert_output_dir.iterdir()):
  logging.warning(f"Output directory ({args.bert_output_dir}) already exists and is not empty!")
args.bert_output_dir.mkdir(parents=True, exist_ok=True)

total_train_examples = 0
for i in range(int(args.num_train_epochs_bert)):
  # The modulo takes into account the fact that we may loop over limited epochs of data
  total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

num_train_optim_steps_bert = int(total_train_examples / args.batch_size_bert / args.gradient_accumulation_steps)
if args.local_rank != -1:
  num_train_optim_steps_bert = num_train_optim_steps_bert // torch.distributed.get_world_size()


## init entailment model optim step

num_line = os.popen( 'wc -l ' + os.path.join(args.qnli_dir,'train_bert_sentiment_file.txt') ).readlines()[0].strip().split()[0]
num_observation_in_train = int(num_line)
print ("\n\nnum_observation_in_train{}".format(num_observation_in_train))

num_train_optim_steps_entailment = int( np.ceil ( np.ceil ( num_observation_in_train / args.batch_size_label ) / args.gradient_accumulation_steps) ) * args.num_train_epochs_entailment + args.batch_size_label

## init joint model

other = {'dropout':0.2}
bert_lm_ent_model = encoder_model.encoder_model (bert_lm_sentence, args, tokenizer, **other )

if args.fp16:
  bert_lm_ent_model.half()

if args.use_cuda:
  bert_lm_ent_model.cuda()

if args.model_load is not None :
  bert_lm_ent_model.load_state_dict(torch.load(args.model_load))

print ('see model to train')
print (bert_lm_ent_model)

## **** train

if not args.not_train: 
  tr_loss = bert_lm_ent_model.train_label(train_label_dataloader,
                                        num_train_optim_steps_entailment,
                                        dev_dataloader=dev_label_dataloader)

  torch.cuda.empty_cache()

  # save
  torch.save(bert_lm_ent_model.state_dict(), os.path.join(args.result_folder,"last_state_dict"+name_add_on+".pytorch"))



train_label_dataloader = None ## clear some space
train_label_examples = None 

print ('\n\nload back best model')
bert_lm_ent_model.load_state_dict( torch.load( os.path.join(args.result_folder,"best_state_dict.pytorch") ) )


""" test set  """

# print ('\n\nload test set')
# processor = data_loader.QnliProcessor()

# dev_label_examples = processor.get_test_examples(args.qnli_dir)
# dev_label_features = data_loader.convert_examples_to_features(dev_label_examples, label_list, MAX_SEQ_LEN, tokenizer)
# dev_label_dataloader = data_loader.make_data_loader (dev_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='sequential')

# print ('\ntest_label_examples {}'.format(len(dev_label_examples))) 

# print ('\n\neval on test')
# result, preds = bert_lm_ent_model.eval_label(dev_label_dataloader)


## **** write vectors 

print ('\n\nwrite vector for each user')

processor = data_loader.LabelProcessorForWrite()
examples = processor.get_examples(args.file_extract_vec_in) ## what vectors do we need to extract ?
features = data_loader.convert_label_desc_to_features(examples,MAX_SEQ_LEN,tokenizer)
features_loader, row_name = data_loader.label_loader_for_write (features,64,fp16=False) ## should take 64 batches easily 


people_emb = bert_lm_ent_model.write_label_vector ( features_loader, args.file_extract_vec_out, row_name)


