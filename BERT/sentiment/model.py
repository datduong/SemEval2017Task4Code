



from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os
from tqdm import tqdm
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

import logging
import json

from scipy.special import softmax

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

from torch.utils.data import DataLoader, Dataset, RandomSampler

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

sys.path.append("/local/datdb/GOmultitask/")

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def convert_example_to_features(example, tokenizer, max_seq_length):
  tokens = example["tokens"]
  segment_ids = example["segment_ids"]
  is_random_next = example["is_random_next"]
  masked_lm_positions = example["masked_lm_positions"]
  masked_lm_labels = example["masked_lm_labels"]

  assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

  input_array = np.zeros(max_seq_length, dtype=np.int)
  input_array[:len(input_ids)] = input_ids

  mask_array = np.zeros(max_seq_length, dtype=np.bool)
  mask_array[:len(input_ids)] = 1

  segment_array = np.zeros(max_seq_length, dtype=np.bool)
  segment_array[:len(segment_ids)] = segment_ids

  lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
  lm_label_array[masked_lm_positions] = masked_label_ids

  features = InputFeatures(input_ids=input_array,
                           input_mask=mask_array,
                           segment_ids=segment_array,
                           lm_label_ids=lm_label_array,
                           is_next=is_random_next)
  return features


class PregeneratedDataset(Dataset):
  def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
    self.vocab = tokenizer.vocab
    self.tokenizer = tokenizer
    self.epoch = epoch
    self.data_epoch = epoch % num_data_epochs
    data_file = training_path / f"epoch_{self.data_epoch}.json"
    metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
    assert data_file.is_file() and metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text())
    num_samples = metrics['num_training_examples']
    seq_len = metrics['max_seq_len']
    self.temp_dir = None
    self.working_dir = None
    if reduce_memory:
      self.temp_dir = TemporaryDirectory()
      self.working_dir = Path(self.temp_dir.name)
      input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
      input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                  shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
      segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                  shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
      lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                   shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
      lm_label_ids[:] = -1
      is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                 shape=(num_samples,), mode='w+', dtype=np.bool)
    else:
      input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
      input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
      segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
      lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
      is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
    logging.info(f"Loading training examples for epoch {epoch}")
    with data_file.open() as f:
      for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
        line = line.strip()
        example = json.loads(line)
        features = convert_example_to_features(example, tokenizer, seq_len)
        input_ids[i] = features.input_ids
        segment_ids[i] = features.segment_ids
        input_masks[i] = features.input_mask
        lm_label_ids[i] = features.lm_label_ids
        is_nexts[i] = features.is_next
    assert i == num_samples - 1  # Assert that the sample count metric was true
    logging.info("Loading complete!")
    self.num_samples = num_samples
    self.seq_len = seq_len
    self.input_ids = input_ids
    self.input_masks = input_masks
    self.segment_ids = segment_ids
    self.lm_label_ids = lm_label_ids
    self.is_nexts = is_nexts

  def __len__(self):
    return self.num_samples

  def __getitem__(self, item):
    return (torch.tensor(self.input_ids[item].astype(np.int64)),
        torch.tensor(self.input_masks[item].astype(np.int64)),
        torch.tensor(self.segment_ids[item].astype(np.int64)),
        torch.tensor(self.lm_label_ids[item].astype(np.int64)),
        torch.tensor(self.is_nexts[item].astype(np.int64)))


def simple_accuracy(preds, labels):
  return (preds == labels).mean()


def acc_and_f1(preds, labels):

  ## do rounding 
  preds = np.round(preds) ## rounding at 0.5 
  acc = simple_accuracy(preds, labels)
  f1 = f1_score(y_true=labels, y_pred=preds)
  return {
      "acc": acc,
      "f1": f1,
      "acc_and_f1": (acc + f1) / 2,
  }


class classify_score (nn.Module):
  def __init__(self, word_vec_dim, out_dim, **kwargs):

    super(classify_score, self).__init__()
    ## take a word vector representing something we want, and do downstream task. 
    ## follow the same layer here
    ## https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py#L1145
    self.LinearLayer = nn.Sequential ( nn.Dropout(kwargs['dropout']), nn.Linear(word_vec_dim,out_dim) ) 
    # self.loss = nn.CrossEntropyLoss()
    self.loss = nn.BCEWithLogitsLoss() ## don't need to call sigmoid, just give it a raw score

  def forward(self,emb1,true_label): 

    score = self.LinearLayer(emb1) ## batch x out_dim (like 16 batch x 5 labels)
    loss = self.loss(score, true_label)
    return loss, score



class encoder_model (nn.Module) :
  def __init__(self,bert_lm_sentence,args,tokenizer,**kwargs):

    super(encoder_model, self).__init__()

    self.tokenizer = tokenizer
    self.bert_lm_sentence = bert_lm_sentence ## bert LM already tuned model
    self.args = args

    self.metric_module = classify_score(768,1, **kwargs) ## okay to hard code, because base bert always give 768, and yelp has 5 labels ... but we split into 2 groups high/low, so prediction is simple yes/no style

  def encode_label_desc (self, label_desc, label_len, label_mask): # @label_desc is matrix row=sentence, col=index
    # encoded_layers , _ = self.bert_lm_sentence.bert (label_desc, output_all_encoded_layers=False) ## do not care about pool

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

    _ , pooled_output = self.bert_lm_sentence.bert (input_ids=label_desc, token_type_ids=None, attention_mask=label_mask, output_all_encoded_layers=False)
    return pooled_output # [batch_size, hidden_size]

  def train_label (self, train_dataloader, num_train_optimization_steps, dev_dataloader=None):

    ## update BERT based on how input-label are matched
    ## BERT.emb is used by words in documents

    param_optimizer = list(self.bert_lm_sentence.bert.named_parameters())  # + list (self.metric_module.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)] , 'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)] + [p for n, p in list(self.metric_module.named_parameters())] , 'weight_decay': 0.0}
      ]

    if self.args.fp16:
      from apex.optimizers import FP16_Optimizer
      from apex.optimizers import FusedAdam

      optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=self.args.learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)

      if self.args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
      else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.args.loss_scale)

      warmup_linear = WarmupLinearSchedule(warmup=self.args.warmup_proportion,
                                            t_total=num_train_optimization_steps)

    else:
      # does not work with --fp16, runs fine with BertAdam
      optimizer = BertAdam(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          warmup=self.args.warmup_proportion,
                          t_total=num_train_optimization_steps)

    self.train()

    global_step = 0
    eval_acc = 0
    last_best_epoch = 0

    for epoch in range( int(self.args.num_train_epochs_entailment)) :

      tr_loss = 0

      for step, batch in enumerate(tqdm(train_dataloader, desc="ent. epoch {}".format(epoch))):
        if self.args.use_cuda:
          batch = tuple(t.cuda() for t in batch)
        else:
          batch = tuple(t for t in batch)

        label_desc1, label_len1, label_mask1, label_ids = batch

        label_desc1.data = label_desc1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
        label_mask1.data = label_mask1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch

        label_emb1 = self.encode_label_desc(label_desc1,label_len1,label_mask1)

        loss, score = self.metric_module.forward(label_emb1, true_label=label_ids)

        if self.args.gradient_accumulation_steps > 1:
          loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
          optimizer.backward(loss)
        else:
          loss.backward()

        tr_loss = tr_loss + loss

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
          if self.args.fp16:
            # modify learning rate with special warm up BERT uses
            # if args.fp16 is False, BertAdam is used that handles this automatically
            lr_this_step = self.args.learning_rate * warmup_linear.get_lr(global_step, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr_this_step

          optimizer.step()
          optimizer.zero_grad()
          global_step += 1

      print ("\ntrain inner epoch {} loss {}".format(epoch,tr_loss))
      # eval at each epoch

      self.eval()

      # print ('\neval on train data inner epoch {}'.format(epoch)) ## too slow, takes 5 mins, we should just skip
      # result, preds = self.eval_label(train_dataloader)

      print ('\neval on dev data inner epoch {}'.format(epoch))
      result, preds = self.eval_label(dev_dataloader)

      if eval_acc < result["acc"]:
        eval_acc = result["acc"] ## better acc
        print ("save best")
        torch.save(self.state_dict(), os.path.join(self.args.result_folder,"best_state_dict.pytorch"))
        last_best_epoch = epoch

      if epoch - last_best_epoch > 20:
        print ('\n\n\n**** break early \n\n\n')
        return tr_loss

      self.train()

    return tr_loss # last train loss

  def eval_label (self, train_dataloader) :

    torch.cuda.empty_cache()

    self.eval()

    preds = []
    all_label_ids = []

    for step, batch in enumerate(tqdm(train_dataloader, desc="eval")):
      if self.args.use_cuda:
        batch = tuple(t.cuda() for t in batch)
      else:
        batch = tuple(t for t in batch)

      with torch.no_grad():

        label_desc1, label_len1, label_mask1, label_ids = batch

        label_desc1.data = label_desc1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
        label_mask1.data = label_mask1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
        label_emb1 = self.encode_label_desc(label_desc1,label_len1,label_mask1)

        loss , prob = self.metric_module.forward(label_emb1, true_label=label_ids)

        prob = F.sigmoid ( prob )

      if len(preds) == 0:
        preds.append(prob.detach().cpu().numpy())
        all_label_ids.append(label_ids.detach().cpu().numpy())
      else:
        preds[0] = np.append(preds[0], prob.detach().cpu().numpy(), axis=0)
        all_label_ids[0] = np.append(all_label_ids[0], label_ids.detach().cpu().numpy(), axis=0) # row array

    # end eval
    all_label_ids = all_label_ids[0]
    preds = preds[0]

    # preds = softmax(preds, axis=1) ## softmax, convert into probability 
  
    print (preds)
    print (all_label_ids)

    result = 0
    if self.args.test_file is None: ## save some time
      result = acc_and_f1(preds, all_label_ids) ## interally, we will take care of the case of @entailment vs @cosine
      for key in sorted(result.keys()):
        print("%s=%s" % (key, str(result[key])))

    return result, preds

  def update_bert (self,num_data_epochs,num_train_optimization_steps):

    ## **** SHOULD NOT DO THIS OFTEN TO AVOID SLOW RUN TIME ****

    ## WITH CURRENT APEX CODE, WE WILL SEE ERROR FOR THE PARAMS NOT USED, ADDED A FIX, SO FOR NOW, WE CAN USE FP16
    ## https://github.com/NVIDIA/apex/issues/131

    param_optimizer = list(self.bert_lm_sentence.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]

    if self.args.fp16:
      try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
      except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

      optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=self.args.learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)


      if self.args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
      else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.args.loss_scale)

      warmup_linear = WarmupLinearSchedule(warmup=self.args.warmup_proportion,
                                            t_total=num_train_optimization_steps)

    else:
      optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=self.args.learning_rate,
                            warmup=self.args.warmup_proportion,
                            t_total=num_train_optimization_steps)

    self.bert_lm_sentence.train()

    global_step = 0

    for epoch in range( int(self.args.num_train_epochs_bert) ) :

      ## call the pregenerated dataset
      epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=self.args.pregenerated_data, tokenizer=self.tokenizer, num_data_epochs=num_data_epochs, reduce_memory=self.args.reduce_memory)
      train_sampler = RandomSampler(epoch_dataset)
      train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=self.args.batch_size_bert)

      ## now do training
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0

      for step, batch in enumerate(tqdm(train_dataloader, desc="bert epoch {}".format(epoch))):
        if self.args.use_cuda:
          batch = tuple(t.cuda() for t in batch)
        else:
          batch = tuple(t for t in batch)

        input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/lm_finetuning/finetune_on_pregenerated.py#L298

        loss = self.bert_lm_sentence(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=lm_label_ids, next_sentence_label=is_next)

        if self.args.gradient_accumulation_steps > 1:
          loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
          optimizer.backward(loss)
        else:
          loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss * self.args.gradient_accumulation_steps / nb_tr_steps

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
          if self.args.fp16:
            # modify learning rate with special warm up BERT uses
            # if args.fp16 is False, BertAdam is used that handles this automatically
            lr_this_step = self.args.learning_rate * warmup_linear.get_lr(global_step, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr_this_step

          optimizer.step()
          optimizer.zero_grad()
          global_step += 1

    return mean_loss

  def second_last_mean (self, label_desc, label_len, label_mask): 
    encoded_layer , _  = self.bert_lm_sentence.bert (input_ids=label_desc, token_type_ids=None, attention_mask=label_mask, output_all_encoded_layers=True)
    second_tolast = encoded_layer[-2]
    second_tolast[label_mask == 0] = 0
    cuda_second_layer = (second_tolast).type(torch.FloatTensor).cuda()
    encode_sum = torch.sum(cuda_second_layer, dim = 1).cuda()
    label_sum = torch.sum(label_mask.cuda(), dim=1).unsqueeze(0).transpose(0,1).type(torch.FloatTensor).cuda()
    go_vectors = encode_sum/label_sum
    return go_vectors

  def write_label_vector (self,label_desc_loader,fout_name,label_name):
    self.eval()

    if fout_name is not None:
      fout = open(fout_name,'w')

    label_emb = None

    counter = 0 ## count the label to be written
    for step, batch in enumerate(tqdm(label_desc_loader, desc="write label desc")):
      if self.args.use_cuda:
        batch = tuple(t.cuda() for t in batch)
      else:
        batch = tuple(t for t in batch)

      label_desc1, label_len1, label_mask1 = batch

      with torch.no_grad():
        label_desc1.data = label_desc1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch
        label_mask1.data = label_mask1.data[ : , 0:int(max(label_len1)) ] # trim down input to max len of the batch

        ## **** USE 2ND LAST AVERAGE TO RETURN THE USER VECTOR
        label_emb1 = self.second_last_mean(label_desc1,label_len1,label_mask1)


      label_emb1 = label_emb1.detach().cpu().numpy()

      if fout_name is not None:
        for row in range ( label_emb1.shape[0] ) :
          fout.write( label_name[counter] + "\t" + "\t".join(str(m) for m in label_emb1[row]) + "\n" )
          counter = counter + 1

      if label_emb is None:
        label_emb = label_emb1
      else:
        label_emb = np.concatenate((label_emb, label_emb1), axis=0) ## so that we have num_go x dim

    if fout_name is not None:
      fout.close()

    return label_emb





