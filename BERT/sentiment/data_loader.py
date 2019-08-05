


## copy the BERT data loader


from __future__ import absolute_import, division, print_function

import argparse, csv, logging, os, random, sys, pickle, gzip, re
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.tokenization import BertTokenizer

sys.path.append("/local/datdb/pytorch-transformers/examples")
import utils_glue as utils_glue ### !!!! NOTICE, THE BERT CODE FROM HUGGINGFACE CHANGED NAME. THEY NOW HAVE THIS OBJECT @utils_glue


logger = logging.getLogger(__name__)

# must match the token correctly. If use BERT tokenizer for labels, then the doctor notes must also use BERT.
# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+') # retain only alphanumeric

def bert_tokenizer_style(tokenizer,text_a, add_cls_sep=True):

  # text_a = re.sub(' the ', " ", text_a) ## can we remove more ? we know "the" is truly not important

  tokens_a = tokenizer.tokenize(text_a)

  ## first sentence
  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0   0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.

  if add_cls_sep:
    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]

  input_1_ids = tokenizer.convert_tokens_to_ids(tokens_a) ## should we use the [CLS] to represent the sentence for downstream task ??

  return input_1_ids , len(input_1_ids)

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_1_ids, input_1_len, input_1_name, input_2_ids=None, input_2_len=None, input_2_name=None , input_ids=None, input_mask=None, segment_ids=None, label_id=None, input_1_mask=None, input_2_mask=None):
    self.input_1_ids = input_1_ids
    self.input_2_ids = input_2_ids
    self.input_1_len = input_1_len
    self.input_2_len = input_2_len
    self.input_1_name = input_1_name
    self.input_2_name = input_2_name
    self.label_id = label_id
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.input_1_mask = input_1_mask
    self.input_2_mask = input_2_mask

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, name_a, text_b=None, name_b=None, label=None, text_len=None):
    """Constructs a InputExample.
    Args:
    guid: Unique id for the example.
    text_a: string. The untokenized text of the first sequence. For single
    sequence tasks, only this sequence must be specified.
    text_b: (Optional) string. The untokenized text of the second sequence.
    Only must be specified for sequence pair tasks.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.name_a = name_a
    self.name_b = name_b
    self.label = label
    self.text_len = text_len


## **** below are needed if we want to extract specific vectors for some sentences

class LabelProcessorForWrite(DataProcessor):
  """Processor for the QNLI data set (GLUE version)."""

  def get_examples(self, label_desc_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(label_desc_dir), "VectorOutput")  # , "train.tsv"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    counter = 0
    examples = []
    for (i, line) in enumerate(lines):
      ## DO NOT HAVE HEADER
      guid = "%s-%s" % (set_type, counter)
      text_a = line[1]
      name_a = line[0] # name are not used, but good for debug
      examples.append(
        InputExample(guid=guid, text_a=text_a.lower(), text_b=None, name_a=name_a, name_b=None, label=None))
      counter = counter + 1
    return examples


def convert_label_desc_to_features(examples, max_seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []

  for (ex_index, example) in tqdm(enumerate(examples)):

    input_1_ids, input_1_len = bert_tokenizer_style(tokenizer, example.text_a, add_cls_sep=False) ## only extract what we need, here CLS is not needed for downstream task
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_1_mask = [1] * len(input_1_ids)
    padding = [0] * (max_seq_length - len(input_1_ids))  # pad zero until max len
    input_1_ids = input_1_ids + padding
    input_1_mask = input_1_mask + padding

    assert len(input_1_ids) == max_seq_length
    assert len(input_1_mask) == max_seq_length

    if ex_index < 5:
      print("\n*** Example ***")
      print("guid: %s" % (example.guid))
      print("tokens: %s" % " ".join([str(x) for x in tokenizer.tokenize(example.text_a)]))
      print("input_ids: %s" % " ".join([str(x) for x in input_1_ids]))
      print("input_mask: %s" % " ".join([str(x) for x in input_1_mask]))


    features.append(InputFeatures(input_1_ids=input_1_ids,
                                  input_1_len=input_1_len,  # true len, not count in the 0-pad
                                  input_1_name=example.name_a,
                                  input_1_mask=input_1_mask
                                  ) )

  return features


def label_loader_for_write (train_features,batch_size,fp16=False):

  all_input_1_name = [f.input_1_name for f in train_features] ## same order as @train_sampler

  all_input_1_ids = torch.tensor([f.input_1_ids for f in train_features], dtype=torch.long)

  all_input_1_len = torch.tensor([f.input_1_len for f in train_features], dtype=torch.float)

  # add segment_ids and input_mask
  all_input_1_mask = torch.tensor([f.input_1_mask for f in train_features], dtype=torch.long)

  all_input_1_ids.data = all_input_1_ids.data[ : , 0:int(max(all_input_1_len)) ] # trim down input to max len of the batch
  all_input_1_mask.data = all_input_1_mask.data[ : , 0:int(max(all_input_1_len)) ] # trim down input to max len of the batch

  if fp16: ## don't need this if we configure BERT fp16 ? well it save some space
    all_input_1_len = all_input_1_len.half() ## 16 bit

  train_data = TensorDataset ( all_input_1_ids, all_input_1_len, all_input_1_mask )

  ## meant for eval data
  train_sampler = SequentialSampler(train_data)

  return DataLoader(train_data, sampler=train_sampler, batch_size=batch_size), all_input_1_name


## **** using glue style code. 

def make_input_data_loader ( args ) :  

  processor = utils_glue.QnliProcessor()
  label_list = processor.get_labels()
  num_labels = len(label_list)

  train_label_examples = processor.get_train_examples(args.qnli_dir) ## testing so use small data 1st

  # examples, label_list, max_seq_length, tokenizer, do_bert_tok=True
  train_label_features = utils_glue.convert_examples_to_features(train_label_examples, label_list, max_seq_length=MAX_SEQ_LEN, tokenizer=tokenizer,output_mode="classification")

  train_label_dataloader = data_loader.make_data_loader (train_label_features,batch_size=args.batch_size_label,fp16=args.fp16, sampler='random')
  print ('\ntrain_label_examples {}'.format(len(train_label_examples))) 
