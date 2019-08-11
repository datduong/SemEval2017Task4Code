
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os, sys, re, pickle
import random

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from tqdm import tqdm, trange


from pytorch_transformers import AdamW, WarmupLinearSchedule

sys.path.append("/local/datdb/pytorch-transformers/examples")
import utils_glue as utils_glue


logger = logging.getLogger(__name__)

from pytorch_transformers.modeling_bert import *


class BertForSequenceClassificationWeighted(BertPreTrainedModel):
  r"""
    **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
      Labels for computing the sequence classification/regression loss.
      Indices should be in ``[0, ..., config.num_labels]``.
      If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
      If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

  Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
    **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
      Classification (or regression if config.num_labels==1) loss.
    **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
      Classification (or regression if config.num_labels==1) scores (before SoftMax).
    **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
      list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
      of shape ``(batch_size, sequence_length, hidden_size)``:
      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    **attentions**: (`optional`, returned when ``config.output_attentions=True``)
      list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

  Examples::

    >>> config = BertConfig.from_pretrained('bert-base-uncased')
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> 
    >>> model = BertForSequenceClassification(config)
    >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    >>> outputs = model(input_ids, labels=labels)
    >>> loss, logits = outputs[:2]

  """
  def __init__(self, config):
    super(BertForSequenceClassificationWeighted, self).__init__(config)
    self.num_labels = config.num_labels

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    self.apply(self.init_weights)

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
        position_ids=None, head_mask=None):
    outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
              attention_mask=attention_mask, head_mask=head_mask)
    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

    if labels is not None:
      if self.num_labels == 1:
        #  We are doing regression
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
      else:
        loss_fct = CrossEntropyLoss(weight=torch.FloatTensor([0.64,2.25]).cuda()) 
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (loss), logits, (hidden_states), (attentions)



## extract vectors from bert model

class Extractor2ndLast (nn.Module):
  def __init__(self,bert_model,args,**kwargs):

    super().__init__()
    self.args = args
    self.bert_model = bert_model
    self.bert_model.bert.encoder.output_hidden_states = True ## turn on this option to see the layers.

  def encode_label_desc (self, label_desc, label_len, label_mask): # @label_desc is matrix row=sentence, col=index

    # # zero padding is not 0, but it has some value, because every character in sentence is "matched" with every other char.
    # # convert padding to actually zero or -inf (if we take maxpool later)
    # encoded_layers.data[label_desc.data==0] = -np.inf ## mask to -inf
    # return mean_sent_encoder ( encoded_layers , label_len ) ## second to last, batch_size x num_word x dim

    ## **** use pooled_output
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L423
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py#L95

    encoded_layer = self.bert_model.bert (input_ids=label_desc, token_type_ids=None, attention_mask=label_mask)
    second_tolast = encoded_layer[2][-2] ## @encoded_layer is tuple in the format: sequence_output, pooled_output, (hidden_states), (attentions)
    second_tolast[label_mask == 0] = 0 ## mask to 0, so that summation over len will not be affected with strange numbers
    cuda_second_layer = (second_tolast).type(torch.FloatTensor).cuda()
    encode_sum = torch.sum(cuda_second_layer, dim = 1).cuda()
    label_sum = torch.sum(label_mask.cuda(), dim=1).unsqueeze(0).transpose(0,1).type(torch.FloatTensor).cuda()
    go_vectors = encode_sum/label_sum
    return go_vectors

  def write_vector (self,label_desc_loader,fout_name,label_name):

    self.eval()

    if fout_name is not None:
      fout = open(fout_name,'w')
      fout.write(str(len(label_name)) + " " + str(768) + "\n") ## based on gensim style, so we can plot it later

    label_emb = None

    counter = 0 ## count the label to be written
    for step, batch in enumerate(tqdm(label_desc_loader, desc="get label vec")):

      batch = tuple(t.cuda() for t in batch)

      label_desc1, label_mask1, label_len1, _ = batch ### input_id, mask, len

      with torch.no_grad():
        label_desc1.data = label_desc1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
        label_mask1.data = label_mask1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
        label_emb1 = self.encode_label_desc(label_desc1,label_len1,label_mask1)

      label_emb1 = label_emb1.detach().cpu().numpy()

      if fout_name is not None:
        for row in range ( label_emb1.shape[0] ) :
          fout.write( label_name[counter] + " " + " ".join(str(m) for m in label_emb1[row]) + "\n" ) ## space, because gensim format
          counter = counter + 1

      if label_emb is None:
        label_emb = label_emb1
      else:
        label_emb = np.concatenate((label_emb, label_emb1), axis=0) ## so that we have num_go x dim

    if fout_name is not None:
      fout.close()

    return label_emb



class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, name=None):
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
    self.label = label
    self.name = name



class LabelDescProcessor(utils_glue.DataProcessor):

  def get_train_examples(self, data_dir, file_name):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, file_name)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "dev.tsv")),
      "dev_matched")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # if i == 0: ## no header
      #   continue
      guid = "%s-%s" % (set_type, line[0])
      text_a = line[1]
      examples.append(
        InputExample(guid=guid, text_a=text_a, label=1, name=line[0])) ## just put @label=1, so we can reuse code
    return examples




class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, input_len, segment_ids, label_id, name=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_len = input_len
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.name = name




def convert_examples_to_features(examples, label_list, max_seq_length,
                 tokenizer, output_mode,
                 cls_token_at_end=False, pad_on_left=False,
                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                 cls_token_segment_id=1, pad_token_segment_id=0,
                 mask_padding_with_zero=True):

  ### **** USE THE SAME FUNCTION AS GITHUB BERT, but we do not add [CLS]

  """ Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
      - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
      - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
  """

  # label_map = {label : i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d of %d" % (ex_index, len(examples)))

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      utils_glue._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
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

    ## !!!  DO NOT ADD [CLS] AND [SEP]

    tokens = tokens_a
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
      tokens += tokens_b
      segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    ## true len
    input_len = len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token] * padding_length) + input_ids
      input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
      segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
      input_ids = input_ids + ([pad_token] * padding_length)
      input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length


    if ex_index < 5:
      print("*** Example ***")
      print("guid: %s" % (example.guid))
      print("tokens: %s" % " ".join(
          [str(x) for x in tokens]))
      print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    features.append(
        InputFeatures(input_ids=input_ids,
                input_mask=input_mask,
                input_len =input_len,
                segment_ids=segment_ids,
                label_id=1,
                name = example.name ))
  return features


def make_loader (args,file_name,tokenizer,batch_size):
  processor = LabelDescProcessor()
  examples = processor.get_train_examples(args.data_dir,file_name)
  features = convert_examples_to_features(examples, None, 512, tokenizer,output_mode=None)

  all_name = [f.name for f in features] ## retain exact ordering as they appear

  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_input_len = torch.tensor([f.input_len for f in features], dtype=torch.float)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

  dataset = TensorDataset(all_input_ids, all_input_mask, all_input_len, all_segment_ids)

  sampler = SequentialSampler(dataset)
  return DataLoader(dataset, sampler=sampler, batch_size=batch_size), all_name


